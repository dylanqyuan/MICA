from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union, List, Tuple

import os
import time
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

from modeling_internvl_chat import InternVLChatModel
from transformers import AutoTokenizer


# -------------------------
# Common constants
# -------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# -------------------------
# InternVL-style preprocess utilities (reused for both scoring + description)
# -------------------------
def build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> List[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images: List[Image.Image] = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images


def load_image(image_file: str, input_size: int = 448, max_num: int = 12) -> torch.Tensor:
    """
    Returns:
      pixel_values: torch.Tensor of shape [tiles, 3, H, W]
    """
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image,
        image_size=input_size,
        use_thumbnail=True,
        max_num=max_num,
    )
    pixel_values = [transform(im) for im in images]
    return torch.stack(pixel_values, dim=0)


# -------------------------
# Public result containers
# -------------------------
@dataclass(frozen=True)
class ScoringResult:
    dist_5: torch.Tensor   # shape [5], CPU float32
    mean_score: float      # expected score in [1,5]
    time_forward: Optional[float] = None


@dataclass(frozen=True)
class DescriptionResult:
    response1: str
    response2: str
    # optional: expose timing if you want to log it later
    time_forward1: Optional[float] = None
    time_forward2: Optional[float] = None


# -------------------------
# Unified Engine
# -------------------------
class MICAEngine:
    """
    Unified inference engine.

    Public API (via dot-operator):
      - mica.scoring(image_path) -> ScoringResult
      - mica.description(image_path) -> DescriptionResult   (two-turn, prompts are fixed)
      - mica.device -> torch.device
    """

    # Write-fixed prompts (your request)
    PROMPT_1 = "Analyze the composition of this image."
    PROMPT_2 = (
        "Classify its dominant composition element into the following categories: "
        "Center, Curved, Diagonal, Horizontal, Pattern, Rule of Thirds, Symmetric, "
        "Triangle, Vertical, golden ratio, fill the frame, vanishing point, or radial."
        "The output format is XXX."
    )

    def __init__(
        self,
        scoring_model: torch.nn.Module,
        description_model: InternVLChatModel,
        description_tokenizer: Any,
        device: torch.device,
        dtype: torch.dtype,
        scoring_input_size: int = 224,
        scoring_max_num: int = 1,
        description_input_size: int = 448,
        description_max_num: int = 1,
        generation_config: Optional[Dict[str, Any]] = None,
        enable_timing: bool = True,
    ) -> None:
        self._scoring_model = scoring_model
        self._desc_model = description_model
        self._desc_tok = description_tokenizer
        self._device = device
        self._dtype = dtype

        # Write-fixed preprocessing knobs
        self._scoring_input_size = scoring_input_size
        self._scoring_max_num = scoring_max_num
        self._description_input_size = description_input_size
        self._description_max_num = description_max_num

        # Default generation config (can still be overridden internally if you later want)
        self._generation_config = generation_config or dict(max_new_tokens=1024, do_sample=False)

        # Timing toggles
        self._enable_timing = enable_timing

    @property
    def device(self) -> torch.device:
        return self._device

    def scoring(self, image_path: str) -> ScoringResult:
        """
        Scoring uses CADBDataset-style preprocessing:
          pixel_values = load_image(image_path, input_size=224, max_num=1).to(bfloat16)
        """
        pixel_values = load_image(
            image_path,
            input_size=self._scoring_input_size,
            max_num=self._scoring_max_num,
        ).to(self._dtype).to(self._device)

        time_forward = None
        if self._enable_timing and self._device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.time()

        pred = self._scoring_model(pixel_values)

        if self._enable_timing and self._device.type == "cuda":
            torch.cuda.synchronize()
        t_end = time.time()
        if self._enable_timing:
            time_forward = t_end - t_start
            
        pred = pred.detach()

        dist_5 = pred.squeeze(0).to("cpu", dtype=torch.float32)  # [5]
        weights = torch.arange(1, 6, dtype=torch.float32)        # [5]
        mean_score = float((dist_5 * weights).sum().item())

        return ScoringResult(dist_5=dist_5, mean_score=mean_score, time_forward=time_forward)

    def description(self, image_path: str) -> DescriptionResult:
        """
        Two-turn fixed-prompt description (your requested API):

          desc_res = mica.description(image_path="demo_imgs/img1.jpg")

        Internals follow your logic:
          - Turn 1: prompt_1, idx_turn=1, history=None  (question prefixed with "<image>\\n")
          - Turn 2: prompt_2, idx_turn=2, history=history1
        Returns response1 and response2.
        """
        # Load pixel_values once (single image multi-turn)
        pixel_values = load_image(
            image_path,
            input_size=self._description_input_size,
            max_num=self._description_max_num,
        ).to(self._dtype).to(self._device)

        # ----- Turn 1 -----
        question1 = "<image>\n" + self.PROMPT_1
        time_forward1 = None
        if self._enable_timing and self._device.type == "cuda":
            torch.cuda.synchronize()
        t1_start = time.time()

        response1, history1 = self._desc_model.chat(
            self._desc_tok,
            pixel_values,
            question1,
            self._generation_config,
            history=None,
            return_history=True,
        )

        if self._enable_timing and self._device.type == "cuda":
            torch.cuda.synchronize()
        t1_end = time.time()
        if self._enable_timing:
            time_forward1 = t1_end - t1_start

        # ----- Turn 2 -----
        question2 = self.PROMPT_2
        time_forward2 = None
        if self._enable_timing and self._device.type == "cuda":
            torch.cuda.synchronize()
        t2_start = time.time()

        response2, history2 = self._desc_model.chat(
            self._desc_tok,
            pixel_values,
            question2,
            self._generation_config,
            history=history1,
            return_history=True,
        )

        if self._enable_timing and self._device.type == "cuda":
            torch.cuda.synchronize()
        t2_end = time.time()
        if self._enable_timing:
            time_forward2 = t2_end - t2_start

        return DescriptionResult(
            response1=response1,
            response2=response2,
            time_forward1=time_forward1,
            time_forward2=time_forward2,
        )


# -------------------------
# Factory: load_mica
# -------------------------
def load_mica(
    *,
    # scoring
    scoring_model_ctor: Callable[[], torch.nn.Module],
    scoring_ckpt_path: str,

    # description
    description_model_path: str,

    # runtime
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.bfloat16,

    # write-fixed knobs
    scoring_input_size: int = 224,
    scoring_max_num: int = 1,
    description_input_size: int = 448,
    description_max_num: int = 1,

    # generation & timing
    generation_config: Optional[Dict[str, Any]] = None,
    enable_timing: bool = True,
) -> MICAEngine:
    """
    Build and return a MICAEngine with:
      - mica.scoring(image_path)
      - mica.description(image_path)  # fixed two-turn prompts

    Scoring preprocessing is fixed to match CADBDataset:
      load_image(image_path, input_size=224, max_num=1).to(bfloat16)
    """
    # Normalize device
    if isinstance(device, str):
        if device == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device(device)

    # 1) Build scoring model
    scoring_model = scoring_model_ctor().to(device).eval()

    if not os.path.exists(scoring_ckpt_path):
        raise FileNotFoundError(f"Scoring checkpoint not found: {scoring_ckpt_path}")

    saved_state_dict = torch.load(scoring_ckpt_path, map_location="cpu")
    model_state_dict = scoring_model.state_dict()
    model_state_dict.update(saved_state_dict)
    scoring_model.load_state_dict(model_state_dict, strict=False)
    scoring_model = scoring_model.to(dtype=dtype)

    # 2) Build description model + tokenizer
    desc_model = InternVLChatModel.from_pretrained(
        description_model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().to(device)

    desc_tok = AutoTokenizer.from_pretrained(
        description_model_path,
        trust_remote_code=True,
        use_fast=False,
    )

    # 3) Wrap into unified engine
    return MICAEngine(
        scoring_model=scoring_model,
        description_model=desc_model,
        description_tokenizer=desc_tok,
        device=device,
        dtype=dtype,
        scoring_input_size=scoring_input_size,
        scoring_max_num=scoring_max_num,
        description_input_size=description_input_size,
        description_max_num=description_max_num,
        generation_config=generation_config,
        enable_timing=enable_timing,
    )
