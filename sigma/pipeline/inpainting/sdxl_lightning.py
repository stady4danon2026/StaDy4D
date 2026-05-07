"""SDXL-Lightning inpainting with IP-Adapter for fast, controlled inference."""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np
import torch
from PIL import Image, ImageOps
from diffusers import AutoPipelineForInpainting, EulerDiscreteScheduler
from transformers import CLIPVisionModelWithProjection

from sigma.data.frame_record import FrameIORecord
from sigma.pipeline.base_stage import StageResult
from sigma.pipeline.inpainting.base_inpainting import BaseInpainter, InpaintingOutputs


# SDXL-Lightning LoRA configurations
SDXL_LIGHTNING_LORAS = {
    "2-step": "ByteDance/SDXL-Lightning",
    "4-step": "ByteDance/SDXL-Lightning",
    "8-step": "ByteDance/SDXL-Lightning",
}

LORA_FILENAMES = {
    "2-step": "sdxl_lightning_2step_lora.safetensors",
    "4-step": "sdxl_lightning_4step_lora.safetensors",
    "8-step": "sdxl_lightning_8step_lora.safetensors",
}


class SDXLLightningInpainter(BaseInpainter):
    """Fast inpainting using SDXL-Lightning with IP-Adapter control.

    SDXL-Lightning is a distilled SDXL model optimized for fast inference,
    supporting 2-step, 4-step, and 8-step generation. This implementation
    combines it with IP-Adapter Plus for reference image conditioning.

    Features:
    - Fast inference with 2/4/8 steps (vs 30-50 for standard SDXL)
    - IP-Adapter Plus for strong reference image control
    - SDXL quality at lightning speed
    """

    name = "sdxl_lightning_inpainter"

    def __init__(
        self,
        num_inference_steps: int = 4,
        enable_ip_adapter: bool = True,
        ip_adapter_config: Dict[str, Any] | None = None,
        ip_adapter_scale: float = 0.6,
        prompt: str = "high quality, realistic background, seamless integration",
        negative_prompt: str = "not consistent with existing pixels, low quality, bad anatomy, distorted, blur, ugly, seams, artifacts",
        device: str = "cuda",
        enable_model_cpu_offload: bool = False,
        **_: Any,
    ) -> None:
        """Initialize SDXL-Lightning inpainter.

        Args:
            num_inference_steps: Number of denoising steps (2, 4, or 8 recommended)
            enable_ip_adapter: Whether to use IP-Adapter for reference conditioning
            ip_adapter_config: Configuration for IP-Adapter
            ip_adapter_scale: Scale for IP-Adapter influence (0.0-1.0)
            prompt: Text prompt for inpainting
            negative_prompt: Negative text prompt
            device: Device to use ("cuda" or "cpu")
            enable_model_cpu_offload: Whether to offload GPU modules to CPU when idle
        """
        # Validate and set inference steps
        if num_inference_steps not in [2, 4, 8]:
            raise ValueError(
                f"num_inference_steps must be 2, 4, or 8 for SDXL-Lightning, got {num_inference_steps}"
            )

        self.num_inference_steps = num_inference_steps
        self.lightning_variant = f"{num_inference_steps}-step"
        self.enable_ip_adapter = enable_ip_adapter
        self.ip_adapter_config = ip_adapter_config or {}
        self.ip_adapter_scale = ip_adapter_scale
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.device = device
        self.enable_model_cpu_offload = enable_model_cpu_offload
        self.pipeline = None
        self.ready = False

    def setup(self) -> None:
        """Load the SDXL inpainting pipeline with Lightning LoRA and IP-Adapter."""
        # Determine dtype based on device
        if self.device == "cuda":
            torch_dtype = torch.float16
            variant = "fp16"
        else:
            torch_dtype = torch.float32
            variant = None

        pipeline_kwargs = {
            "torch_dtype": torch_dtype,
        }

        # Handle Image Encoder for IP-Adapter (must be passed to from_pretrained)
        image_encoder = None
        if self.enable_ip_adapter:
            image_encoder_repo = self.ip_adapter_config.get("image_encoder_repo")
            image_encoder_subfolder = self.ip_adapter_config.get("image_encoder_subfolder")

            if image_encoder_repo:
                encoder_kwargs = {"subfolder": image_encoder_subfolder} if image_encoder_subfolder else {}
                image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    image_encoder_repo,
                    **encoder_kwargs
                ).to(self.device, dtype=torch_dtype)
                pipeline_kwargs["image_encoder"] = image_encoder

        if variant:
            pipeline_kwargs["variant"] = variant

        # Load base SDXL inpainting pipeline
        self.pipeline = AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            **pipeline_kwargs
        )

        # Load SDXL-Lightning LoRA (UNet only - no text encoder weights)
        lora_repo = SDXL_LIGHTNING_LORAS[self.lightning_variant]
        lora_filename = LORA_FILENAMES[self.lightning_variant]

        self.pipeline.load_lora_weights(
            lora_repo,
            weight_name=lora_filename
        )
        # Fuse LoRA for faster inference as recommended by SDXL-Lightning docs
        if hasattr(self.pipeline, "fuse_lora"):
            self.pipeline.fuse_lora()

        # Configure scheduler for SDXL-Lightning (EulerDiscreteScheduler recommended)
        self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
            self.pipeline.scheduler.config,
            timestep_spacing="trailing"
        )

        # Load IP-Adapter if enabled
        if self.enable_ip_adapter:
            repo = self.ip_adapter_config.get("repo", "h94/IP-Adapter")
            subfolder = self.ip_adapter_config.get("subfolder", "sdxl_models")
            weight_name = self.ip_adapter_config.get("weight_name", "ip-adapter-plus_sdxl_vit-h.bin")

            load_kwargs = {
                "subfolder": subfolder,
                "weight_name": weight_name,
            }

            self.pipeline.load_ip_adapter(repo, **load_kwargs)
            self.pipeline.set_ip_adapter_scale(self.ip_adapter_scale)

        # Move to device or use CPU offload
        if self.device == "cuda":
            if self.enable_model_cpu_offload:
                self.pipeline.enable_model_cpu_offload()
            else:
                self.pipeline.to(self.device)
        else:
            # For CPU, just move to device
            self.pipeline.to(self.device)

        self.pipeline.set_progress_bar_config(disable=True)
        self.ready = True

    def _preprocess_image(self, image: np.ndarray, target_size: int = 1024) -> Image.Image:
        """Convert numpy array to PIL Image and resize preserving aspect ratio."""
        pil_image = Image.fromarray(image)

        # Resize to be compatible with SDXL (1024x1024 native)
        # We resize the short side to target_size
        w, h = pil_image.size
        scale = target_size / min(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        # Ensure dimensions are divisible by 8 (VAE requirement)
        new_w = (new_w // 16) * 16
        new_h = (new_h // 16) * 16

        return pil_image.resize((new_w, new_h), Image.LANCZOS)

    def _preprocess_mask(self, mask: np.ndarray, target_size: tuple[int, int]) -> Image.Image:
        """Convert numpy mask to PIL Image and resize."""
        # Mask is expected to be 0 (keep) and 1 (inpaint) or 255 (inpaint)
        # Ensure it's uint8 0-255
        if mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)

        pil_mask = Image.fromarray(mask)
        return pil_mask.resize(target_size, Image.NEAREST)

    def process(self, frame_records: Dict[int, FrameIORecord], frame_idx: int) -> StageResult:
        """Invoke SDXL-Lightning inpainting with IP-Adapter control.

        Args:
            frame_records: Stored frames keyed by index.
            frame_idx: Current timestep to inpaint.

        Returns:
            StageResult bundling the inpainted image.
        """
        if not self.ready or self.pipeline is None:
            raise RuntimeError("SDXLLightningInpainter.setup() must run before process().")

        curr_record = frame_records.get(frame_idx)
        prev_record = frame_records.get(frame_idx - 1)
        if curr_record is None:
            raise ValueError(f"Frame record for index {frame_idx} is required.")

        input_img_np = curr_record.origin_image
        if input_img_np is None:
            raise ValueError("No input image found for current frame.")

        mask_np = curr_record.mask
        if mask_np is None:
            # If no mask, return original image (nothing to inpaint)
            outputs = InpaintingOutputs(inpainted_image=input_img_np, confidence_map=None)
            curr_record.inpainted_image = outputs.inpainted_image
            return StageResult(data={"inpainting": outputs}, visualization_assets={})

        ref_img_np = None
        if prev_record is not None:
            ref_img_np = prev_record.inpainted_image if prev_record.inpainted_image is not None else prev_record.origin_image

        # Preprocess
        pil_input = self._preprocess_image(input_img_np)
        pil_mask = self._preprocess_mask(mask_np, pil_input.size)
        # pil_mask = ImageOps.invert(pil_mask)

        ip_adapter_image = None
        if self.enable_ip_adapter and ref_img_np is not None:
            # IP-Adapter expects PIL images
            ref_img_rgb = ref_img_np
            ip_adapter_image = Image.fromarray(ref_img_rgb)

        # Inference with SDXL-Lightning (fast - only 2/4/8 steps)
        generator_device = "cpu" if self.device == "cpu" else "cuda"
        generator = torch.Generator(device=generator_device).manual_seed(42)

        # Handle IP-Adapter when no reference image is available (e.g. first frame)
        current_scale = self.ip_adapter_scale
        if self.enable_ip_adapter and ip_adapter_image is None:
            # Set scale to 0 to ignore the dummy image
            self.pipeline.set_ip_adapter_scale(0.0)
            # Create a dummy image to satisfy API requirements
            ip_adapter_image = Image.new("RGB", (224, 224), (0, 0, 0))

        try:
            # SDXL-Lightning specific parameters
            # - guidance_scale: 0 for distilled models (CFG-free)
            # - num_inference_steps: must match the LoRA variant (2/4/8)
            output = self.pipeline(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                image=pil_input,
                mask_image=pil_mask,
                ip_adapter_image=ip_adapter_image if self.enable_ip_adapter else None,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=0,  # SDXL-Lightning uses CFG-free distillation
                generator=generator,
            ).images[0]
        finally:
            # Restore scale
            if self.enable_ip_adapter:
                self.pipeline.set_ip_adapter_scale(current_scale)

        # Post-process: resize back to original resolution
        orig_h, orig_w = input_img_np.shape[:2]
        output_np = np.array(output.resize((orig_w, orig_h), Image.LANCZOS))

        # Keep as RGB
        output_rgb = output_np

        # Create visualization assets
        resized_input_np = np.array(pil_input)
        resized_mask_np = np.array(pil_mask)

        # Create overlay for visualization (red tint on mask)
        overlay = resized_input_np.copy()
        mask_indices = resized_mask_np > 127
        if mask_indices.any():
            overlay[mask_indices] = (overlay[mask_indices] * 0.6 + np.array([255, 0, 0]) * 0.4).astype(np.uint8)

        outputs = InpaintingOutputs(inpainted_image=output_rgb, confidence_map=None)
        curr_record.inpainted_image = output_rgb

        vis_assets = {
            "inpainted_image": output_rgb,
            "model_name": f"SDXL-Lightning ({self.lightning_variant})",
            "prompt": self.prompt,
            "conditioning_image": ref_img_np,
            "overlay": overlay,
            "resized_input": resized_input_np,
            "resized_mask": resized_mask_np,
            "num_inference_steps": self.num_inference_steps,
        }

        return StageResult(data={"inpainting": outputs}, visualization_assets=vis_assets)

    def teardown(self) -> None:
        """Clean up resources."""
        self.pipeline = None
        self.ready = False
