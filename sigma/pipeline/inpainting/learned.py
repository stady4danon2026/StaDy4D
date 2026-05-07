"""Learning-based inpainting using SDXL and IP-Adapter."""

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from diffusers import AutoPipelineForInpainting
from transformers import CLIPVisionModelWithProjection

from sigma.data.frame_record import FrameIORecord
from sigma.pipeline.base_stage import StageResult
from sigma.pipeline.inpainting.base_inpainting import BaseInpainter, InpaintingOutputs


class LearnedInpainter(BaseInpainter):
    """Inpainting stage using SDXL with optional IP-Adapter control."""

    name = "learned_inpainter"

    def __init__(
        self,
        model_name: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        enable_ip_adapter: bool = True,
        ip_adapter_config: Dict[str, Any] | None = None,
        ip_adapter_scale: float = 0.6,
        prompt: str = "high quality, realistic background, seamless integration",
        negative_prompt: str = "not consistent with existing pixels, low quality, bad anatomy, distorted, blur, ugly, seams, artifacts",
        device: str = "cuda",
        enable_model_cpu_offload: bool = False,
        **_: Any,
    ) -> None:
        self.model_name = model_name
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
        """Load the SDXL pipeline and IP-Adapter."""
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
        
        # Handle Image Encoder for SDXL (must be passed to from_pretrained)
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

        self.pipeline = AutoPipelineForInpainting.from_pretrained(
            self.model_name,
            **pipeline_kwargs
        )

        # Load IP-Adapter if enabled
        if self.enable_ip_adapter:
            repo = self.ip_adapter_config.get("repo", "h94/IP-Adapter")
            subfolder = self.ip_adapter_config.get("subfolder", "sdxl_models")
            weight_name = self.ip_adapter_config.get("weight_name", "ip-adapter_sdxl.bin")
            
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
        """Invoke Inpainting (SDXL) with IP-Adapter control.

        Args:
            frame_records: Stored frames keyed by index.
            frame_idx: Current timestep to inpaint.

        Returns:
            StageResult bundling the inpainted image.
        """
        if not self.ready or self.pipeline is None:
            raise RuntimeError("LearnedInpainter.setup() must run before process().")

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

        # 2. Preprocess
        # SDXL works best at ~1024x1024.
        pil_input = self._preprocess_image(input_img_np)
        pil_mask = self._preprocess_mask(mask_np, pil_input.size)
        # pil_mask = ImageOps.invert(pil_mask)
        
        ip_adapter_image = None
        if self.enable_ip_adapter and ref_img_np is not None:
            # IP-Adapter expects the reference image to be processed by CLIP image processor
            # The pipeline handles PIL images directly.
            # Input is already RGB
            ref_img_rgb = ref_img_np
            ip_adapter_image = Image.fromarray(ref_img_rgb)

        # 3. Inference
        # Use appropriate device for generator
        generator_device = "cpu" if self.device == "cpu" else "cuda"
        generator = torch.Generator(device=generator_device).manual_seed(42)  # For reproducibility
        
        # Handle IP-Adapter when no reference image is available (e.g. first frame)
        current_scale = self.ip_adapter_scale
        if self.enable_ip_adapter and ip_adapter_image is None:
            # Set scale to 0 to ignore the dummy image
            self.pipeline.set_ip_adapter_scale(0.0)
            # Create a dummy image to satisfy API requirements
            ip_adapter_image = Image.new("RGB", (224, 224), (0, 0, 0))
        
        try:
            output = self.pipeline(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                image=pil_input,
                mask_image=pil_mask,
                ip_adapter_image=ip_adapter_image if self.enable_ip_adapter else None,
                num_inference_steps=30,
                strength=0.99,
                guidance_scale=7.5,
                generator=generator,
            ).images[0]
        finally:
            # Restore scale
            if self.enable_ip_adapter:
                self.pipeline.set_ip_adapter_scale(current_scale)

        # 4. Post-process
        # Resize back to original resolution
        orig_h, orig_w = input_img_np.shape[:2]
        output_np = np.array(output.resize((orig_w, orig_h), Image.LANCZOS))
        
        # Keep as RGB (don't convert to BGR)
        output_rgb = output_np

        # 5. Return
        resized_input_np = np.array(pil_input)
        resized_mask_np = np.array(pil_mask)
        
        # Create overlay for visualization (red tint on mask)
        overlay = resized_input_np.copy()
        # Assuming pil_mask is the mask passed to model. 
        # Visualize high values as red.
        mask_indices = resized_mask_np > 127
        if mask_indices.any():
            overlay[mask_indices] = (overlay[mask_indices] * 0.6 + np.array([255, 0, 0]) * 0.4).astype(np.uint8)

        outputs = InpaintingOutputs(inpainted_image=output_rgb, confidence_map=None)
        curr_record.inpainted_image = output_rgb
        vis_assets = {
            "inpainted_image": output_rgb, 
            "model_name": self.model_name,
            "prompt": self.prompt,
            "conditioning_image": ref_img_np,
            "overlay": overlay,
            "resized_input": resized_input_np,
            "resized_mask": resized_mask_np
        }
        return StageResult(data={"inpainting": outputs}, visualization_assets=vis_assets)

    def teardown(self) -> None:
        self.pipeline = None
        self.ready = False


if __name__ == '__main__':
    # imports for testing
    import os

    # Initialize a LearnedInpainter instance
    inpainter = LearnedInpainter(enable_ip_adapter=True)
    inpainter.setup()
    
    # Load an image for experimenting
    # Ensure these paths exist or use dummy data if running purely for code check
    if os.path.exists('outputs/visuals/inpainting/frame_00001/inpainting_conditioning_image.png'):
        frame0 = cv2.cvtColor(cv2.imread('demo/data/frames/frame_00000.jpg'), cv2.COLOR_BGR2RGB)
        frame1 = cv2.cvtColor(cv2.imread('demo/data/frames/frame_00020.jpg'), cv2.COLOR_BGR2RGB)
    else:
        # Dummy data
        print("Warning: Demo images not found. Using dummy black images.")
        frame0 = np.zeros((512, 512, 3), dtype=np.uint8)
        frame1 = np.zeros((512, 512, 3), dtype=np.uint8)

    # Setup
    output_root = 'outputs/test___'
    os.makedirs(output_root, exist_ok=True)

    # Create a mask
    mask = np.zeros(frame1.shape[:2], dtype=np.uint8)
    
    use_random_mask = True
    if use_random_mask:
        # Create a mask with multiple random blocks
        num_blocks = 30
        np.random.seed(42) # For reproducibility
        for _ in range(num_blocks):
            w = np.random.randint(30, 300)
            h = np.random.randint(30, 300)
            x = np.random.randint(0, frame1.shape[1] - w)
            y = np.random.randint(0, frame1.shape[0] - h)
            cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
    else:
        # Create a mask by specifying x, y, w, h
        x, y, w, h = 490, 740, (510-490), (770-740)
        cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)

    # Save inputs
    # Save inputs (convert RGB back to BGR for cv2)
    cv2.imwrite(f'{output_root}/original.jpg', cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f'{output_root}/control.jpg', cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f'{output_root}/input_masked.jpg', cv2.cvtColor(cv2.bitwise_and(frame1, frame1, mask=mask), cv2.COLOR_RGB2BGR))

    # Create frame records for the demo call
    frame_records = {
        0: FrameIORecord(frame_idx=0, origin_image=frame0),
        1: FrameIORecord(frame_idx=1, origin_image=frame1, mask=mask),
    }

    # Process the inputs
    try:
        print("Running inpainting process...")
        result = inpainter.process(frame_records, 1)
        output_frame = frame_records[1].inpainted_image
        # Save the inpainted image
        if output_frame is not None:
            cv2.imwrite(f'{output_root}/inpainted_frame_00000.jpg', cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
            print(f"Inpainting successful. Output saved to {output_root}/inpainted_frame_00000.jpg")
        else:
            print("Inpainting returned None.")
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        inpainter.teardown()
