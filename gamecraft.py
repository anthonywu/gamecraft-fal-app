import json
import fal
from fal.container import ContainerImage
from fal.toolkit import (
    File as FalFile,
    Image as FalImage,
    Video as FalVideo,  # noqa
)
from pydantic import BaseModel, Field, ValidationError


class ModelWarmupError(Exception): ...


class Input(BaseModel):
    image: FalImage = Field(description="The source image to generate a video from.")
    prompt: str = Field(
        description="The source image's companion prompt to generate the video from",
        examples=[
            "A charming Parisian street scene with its vibrant red awning and outdoor seating area, surrounded by quant shops and lush greenery under a bright blue sky."
        ],
    )
    neg_prompt: str = Field(
        description="The source image to generate a video from.",
        default="overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border.",
    )
    action_list: str = Field(
        description="JSON list of camera actions: w=forward, a=left, d=right, s=backward",
        default=json.dumps(["w", "s", "d", "a"]),
        examples=[
            json.dumps(["w", "w", "w", "a", "a", "a"]),  # go forward 3x, turn left 3x
            json.dumps(["d", "d", "d", "d"]),  # pull back camera
        ],
    )
    action_speed_list: str = Field(
        description="JSON list of speeds to match action_list. Must be same length as action_list.",
        default=json.dumps([0.2, 0.2, 0.2, 0.2]),
        examples=[
            json.dumps([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),  # 6 actions
            json.dumps([0.1, 0.2, 0.3, 0.4]),  # pull back camera from slow to fast
        ],
    )
    cfg_scale: float = Field(description="CFG Scale", default=2.0)
    infer_steps: int = Field(description="Number of Inference Steps", default=5)
    seed: int = Field(description="Generation Seed", default=42)

    def parse_action_fields(self):
        a_list = json.loads(self.action_list)
        s_list = json.loads(self.action_speed_list)
        if len(a_list) != len(s_list):
            raise ValidationError(
                "action list and action speed list must be the same length"
            )
        if not all([_ in ["w", "s", "d", "a"] for _ in a_list]):
            raise ValidationError("action list values must be be one of w s d a")
        if not all([isinstance(_, (int, float)) for _ in s_list]):
            raise ValidationError("action list values must be be one of int or float")
        return a_list, s_list


class Output(BaseModel):
    # example: https://docs.fal.ai/serverless/tutorials/deploy-text-to-video-model
    video: FalFile = Field(
        description="The generated Gamecraft video file",
    )


class FalGamecraftModel(
    fal.App,
    name="gamecraft",
    image=ContainerImage.from_dockerfile("./Dockerfile"),
    kind="container",
    keep_alive=300,
    min_concurrency=0,  # Scale to zero when idle
    max_concurrency=2,  # Limit concurrent requests
):
    machine_type = "GPU-H100"

    def setup(self) -> None:
        import os
        import subprocess
        from pathlib import Path

        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        import torch

        print("Device capability:", torch.cuda.get_device_capability())
        print("CUDA version:", torch.version.cuda)
        world_size = torch.cuda.device_count()
        print(f"Found {world_size} GPUs.")

        import triton

        print("Triton version:", triton.__version__)

        self.supports_fp8 = False  # torch.cuda.get_device_capability() >= (8, 9)

        # Check disk space before attempting to download anything
        print("--- Checking disk space before download ---")
        try:
            result = subprocess.run(
                ["df", "-h"], check=True, capture_output=True, text=True
            )
            print(result.stdout)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Could not check disk space: {e}")
        print("-------------------------------------------")

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
        os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "1000000000000"
        os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = "32"

        """
        Expected layout of /data/Hunyuan-GameCraft-1.0
        ├──weights
        │  ├──gamecraft_models
        │  │  │──mp_rank_00_model_states.pt
        │  │  │──mp_rank_00_model_states_distill.pt
        │  │──stdmodels
        │  │  ├──vae_3d
        │  │  │  │──hyvae
        │  │  │  │  ├──pytorch_model.pt
        │  │  │  │  ├──config.json
        │  │  ├──llava-llama-3-8b-v1_1-transformers
        │  │  │  ├──model-00001-of-00004.safatensors
        │  │  │  ├──model-00002-of-00004.safatensors
        │  │  │  ├──model-00003-of-00004.safatensors
        │  │  │  ├──model-00004-of-00004.safatensors
        │  │  │  ├──...
        │  │  ├──openai_clip-vit-large-patch14
        """

        from huggingface_hub import snapshot_download

        weights_parent_dir = Path("/data/Hunyuan-GameCraft-1.0")
        if not Path("/data/Hunyuan-GameCraft-1.0").exists():
            snapshot_download(
                repo_id="tencent/Hunyuan-GameCraft-1.0", local_dir=weights_parent_dir
            )

        self.model_base_dir = weights_parent_dir / "stdmodels"
        self.checkpoint_path = (
            weights_parent_dir / "gamecraft_models/mp_rank_00_model_states_distill.pt"
        )
        print(f"Weights download confirmed, weights dir at {weights_parent_dir}")

        self.video_size = ["704", "1216"]
        self.save_path = "/data/gamecraft-output"
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        self.script = "/opt/gamecraft/hymm_sp/sample_batch.py"

        print("Warming up model with first run")
        warmup_output_path = self._torch_run(
            Input(
                image=FalImage.from_path(
                    "/data/sample-images/unsc-soldier.jpg"
                    # "/opt/gamecraft/sample-input-parisian-street.png"  - Hunyuan example
                ),
                prompt="Halo UNSC soldier prepares weapon and squares up for imminent battle with space aliens",
                # prompt="A charming Parisian street scene with its vibrant red awning and outdoor seating area, surrounded by quant shops and lush greenery under a bright blue sky.",
            )
        )
        print(f"Warming up model with first run, output: {warmup_output_path}")

    def _torch_run(self, input: Input):
        import os
        from datetime import datetime
        from pathlib import Path

        os.environ["PYTHONPATH"] = "/opt/gamecraft"
        os.environ["MODEL_BASE"] = str(self.model_base_dir)
        os.environ["DISABLE_SP"] = "1"
        os.environ["CPU_OFFLOAD"] = "1"
        os.environ["NUM_GPU"] = "1"

        timestamp = datetime.now().timestamp()
        microseconds = int(timestamp * 1_000_000)

        # we need to give the gamescript script a image path
        image_save_path = Path(self.save_path) / f"{microseconds}.png"
        input.image.save(
            image_save_path, overwrite=True
        )  # the script expects a file not a image wrapper

        # for now the hunyuan batch script just renames to image basename + .mp4
        expected_mp4_path = image_save_path.with_suffix(".mp4")

        # we use a SimpleNamespace to substite the expected interface to sample_batch.py's main()
        # without refactoring *upstream code* right now
        from types import SimpleNamespace

        action_list, action_speed_list = input.parse_action_fields()
        worker_args = SimpleNamespace(
            image_path=str(image_save_path),
            image_start=True,  # vs. video_start
            prompt=input.prompt,
            add_neg_prompt=input.neg_prompt,
            infer_steps=input.infer_steps,
            ckpt=str(self.checkpoint_path),
            cfg_scale=input.cfg_scale,
            cpu_offload=False,
            seed=input.seed,
            action_list=action_list,
            action_speed_list=action_speed_list,
            flow_shift_eval_video=5.0,
            use_sage=False,
            use_deepcache=1,
            sample_n_frames=33,
            use_linear_quadratic_schedule=True,
            linear_schedule_end=25,
            save_path=self.save_path,
            video_path=None,
            save_path_suffix="",
            precision="bf16",  # Added precision attribute
            num_images=1,  # Added num_images attribute that may also be needed
            latent_channels=16,  # Let it be determined by VAE model
            rope_theta=256,
            model="HYVideo-T/2",
            vae="884-16c-hy0801",
            vae_precision="fp16",
            vae_tiling=True,
            text_encoder="llava-llama-3-8b",
            text_encoder_precision="fp16",
            text_states_dim=4096,
            text_len=256,
            tokenizer="llava-llama-3-8b",
            text_encoder_infer_mode="encoder",
            prompt_template_video="li-dit-encode-video",
            hidden_state_skip_layer=2,
            apply_final_norm=True,
            text_encoder_2="clipL",
            text_encoder_precision_2="fp16",
            text_states_dim_2=768,
            text_len_2=77,
            tokenizer_2="clipL",
            text_projection="single_refiner",
            flow_reverse=True,
            flow_solver="euler",
            use_attention_mask=True,  # Added for HYVideoDiffusionTransformer
            load_key="module",  # Key for loading state dict
            use_fp8=self.supports_fp8,
            reproduce=True,  # Enable reproducible generation
            ip_cfg_scale=0,
            val_disable_autocast=False,
        )

        try:
            # Run inference directly without multiprocessing spawn
            # Since we're already in a containerized environment with dedicated GPU,
            # we don't need the additional multiprocessing layer
            inference_main(0, worker_args)
        except Exception as e:
            print("An error occurred during gamecraft inference!")
            print(f"Error: {e}")
            if hasattr(e, "cmd"):
                print(
                    f"Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
                )
            if hasattr(e, "stdout"):
                print("\n--- STDOUT ---")
                print(e.stdout)
            if hasattr(e, "stderr"):
                print("\n--- STDERR ---")
                print(e.stderr)
            import traceback

            traceback.print_exc()
            raise ModelWarmupError(
                "Unable to warm up the model with test run, aborting job"
            ) from e

        return expected_mp4_path

    @fal.endpoint("/")
    def image_to_video(self, input: Input) -> Output:
        expected_mp4_path = self._torch_run(input)
        return Output(video=FalFile.from_path(expected_mp4_path))


# adapted code from sample_batch.py - keeping only image -> video and cut out video -> video for now


class CropResize:
    """
    Custom transform to resize and crop images to a target size while preserving aspect ratio.

    Resizes the image to ensure it covers the target dimensions, then center-crops to the exact size.
    Useful for preparing consistent input dimensions for video generation models.
    """

    def __init__(self, size=(704, 1216)):
        """
        Args:
            size (tuple): Target dimensions (height, width) for the output image
        """
        self.target_h, self.target_w = size

    def __call__(self, img):
        """
        Apply the transform to an image.

        Args:
            img (PIL.Image): Input image to transform

        Returns:
            PIL.Image: Resized and cropped image with target dimensions
        """
        import torchvision.transforms as transforms

        # Get original image dimensions
        w, h = img.size

        # Calculate scaling factor to ensure image covers target size
        scale = max(
            self.target_w / w,  # Scale needed to cover target width
            self.target_h / h,  # Scale needed to cover target height
        )

        # Resize image while preserving aspect ratio
        new_size = (int(h * scale), int(w * scale))
        resize_transform = transforms.Resize(
            new_size, interpolation=transforms.InterpolationMode.BILINEAR
        )
        resized_img = resize_transform(img)

        # Center-crop to exact target dimensions
        crop_transform = transforms.CenterCrop((self.target_h, self.target_w))
        return crop_transform(resized_img)


def inference_main(rank, worker_args):
    """
    Main function for video generation using the Hunyuan multimodal model.

    Handles argument parsing, distributed setup, model loading, data preparation,
    and video generation with action-controlled transitions. Supports both image-to-video
    and video-to-video generation tasks.
    """
    import os
    import random
    from pathlib import Path
    from loguru import logger

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    import torch
    import torch.distributed
    import torchvision.transforms as transforms
    from PIL import Image

    from hymm_sp.sample_inference import HunyuanVideoSampler
    from hymm_sp.data_kits.data_tools import save_videos_grid

    # Parse command-line arguments and configuration
    args = worker_args
    models_root_path = Path(args.ckpt)
    action_list = args.action_list
    action_speed_list = args.action_speed_list
    negative_prompt = args.add_neg_prompt

    # Set random seed for reproducibility
    logger.info("*" * 20)
    torch.manual_seed(args.seed)

    # Validate model checkpoint path exists
    if not models_root_path.exists():
        raise ValueError(f"Model checkpoint path does not exist: {models_root_path}")
    logger.info("+" * 20)

    # Set up output directory
    save_path = (
        args.save_path
        if args.save_path_suffix == ""
        else f"{args.save_path}_{args.save_path_suffix}"
    )
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"Generated videos will be saved to: {save_path}")

    # Initialize device configuration for single GPU
    device = torch.device("cuda")

    # Load the Hunyuan video sampler model from checkpoint
    logger.info(f"Loading model from checkpoint: {args.ckpt}")
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(
        args.ckpt,
        args=args,
        device=device if not args.cpu_offload else torch.device("cpu"),
    )
    hunyuan_video_sampler.ip_cfg_scale = 0  # hack arg expected but not parsed??
    # Update args with model-specific configurations from the checkpoint
    args = hunyuan_video_sampler.args

    # Enable CPU offloading if specified to reduce GPU memory usage
    if args.cpu_offload:
        from diffusers.hooks import apply_group_offloading

        onload_device = torch.device("cuda")
        apply_group_offloading(
            hunyuan_video_sampler.pipeline.transformer,
            onload_device=onload_device,
            offload_type="block_level",
            num_blocks_per_group=1,
        )
        logger.info("Enabled CPU offloading for transformer blocks")

    # Process each batch in the dataset

    prompt = args.prompt
    image_paths = [args.image_path]
    logger.info(f"Prompt: {prompt}, Image Path {args.image_path}")
    # Generate random seed for reproducibility
    seed = args.seed if args.seed else random.randint(0, 1_000_000)

    # Define image transformation pipeline for input reference images
    closest_size = (704, 1216)
    ref_image_transform = transforms.Compose(
        [
            CropResize(closest_size),
            transforms.CenterCrop(closest_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1] range
        ]
    )

    # Handle image-based generation (start from a single image)
    # Load and preprocess reference images
    raw_ref_images = [
        Image.open(image_path).convert("RGB") for image_path in image_paths
    ]

    # Apply transformations and prepare tensor for model input
    ref_images_pixel_values = [
        ref_image_transform(ref_image) for ref_image in raw_ref_images
    ]
    ref_images_pixel_values = (
        torch.cat(ref_images_pixel_values).unsqueeze(0).unsqueeze(2).to(device)
    )

    # Encode reference images to latent space using VAE
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        if args.cpu_offload:
            # Move VAE components to GPU temporarily for encoding
            hunyuan_video_sampler.vae.quant_conv.to("cuda")
            hunyuan_video_sampler.vae.encoder.to("cuda")

        # Enable tiling for VAE to handle large images efficiently
        hunyuan_video_sampler.pipeline.vae.enable_tiling()

        # Encode image to latents and scale by VAE's scaling factor
        raw_last_latents = (
            hunyuan_video_sampler.vae.encode(ref_images_pixel_values)
            .latent_dist.sample()
            .to(dtype=torch.float16)
        )  # Shape: (B, C, F, H, W)
        raw_last_latents.mul_(hunyuan_video_sampler.vae.config.scaling_factor)
        raw_ref_latents = raw_last_latents.clone()

        # Clean up
        hunyuan_video_sampler.pipeline.vae.disable_tiling()
        if args.cpu_offload:
            # Move VAE components back to CPU after encoding
            hunyuan_video_sampler.vae.quant_conv.to("cpu")
            hunyuan_video_sampler.vae.encoder.to("cpu")

    # Store references for generation loop
    ref_images = raw_ref_images
    last_latents = raw_last_latents
    ref_latents = raw_ref_latents

    # Generate video segments for each action in the action list
    for idx, action_id in enumerate(action_list):
        # Determine if this is the first action and using image start
        is_image = idx == 0 and args.image_start

        logger.info(
            f"Generating segment {idx + 1}/{len(action_list)} with action ID: {action_id}"
        )
        # Generate video segment with the current action
        outputs = hunyuan_video_sampler.predict(
            prompt=prompt,
            action_id=action_id,
            action_speed=action_speed_list[idx],
            is_image=is_image,
            size=(704, 1216),
            seed=seed,
            last_latents=last_latents,  # Previous frame latents for continuity
            ref_latents=ref_latents,  # Reference latents for style consistency
            video_length=args.sample_n_frames,
            guidance_scale=args.cfg_scale,
            ip_cfg_scale=0,
            num_images_per_prompt=args.num_images,
            negative_prompt=negative_prompt,
            infer_steps=args.infer_steps,
            flow_shift=args.flow_shift_eval_video,
            use_linear_quadratic_schedule=args.use_linear_quadratic_schedule,
            linear_schedule_end=args.linear_schedule_end,
            use_deepcache=args.use_deepcache,
            cpu_offload=args.cpu_offload,
            ref_images=ref_images,
            output_dir=save_path,
            return_latents=True,
            use_sage=args.use_sage,
            precision="bf16",
            reproduce=True,
            load_key="module",
            use_fp8=args.use_fp8,
            use_attention_mask=args.use_attention_mask,
        )

        # Update latents for next iteration (maintain temporal consistency)
        ref_latents = outputs["ref_latents"]
        last_latents = outputs["last_latents"]

        # Save generated video segments if this is the main process (rank 0)
        if rank == 0:
            sub_samples = outputs["samples"][0]

            # Initialize or concatenate video segments
            if idx == 0:
                if args.image_start:
                    out_cat = sub_samples
            else:
                # Append new segment to existing video
                out_cat = torch.cat([out_cat, sub_samples], dim=2)

            # Save final combined video
            save_path_mp4 = (
                f"{save_path}/{os.path.basename(args.image_path).split('.')[0]}.mp4"
            )
            save_videos_grid(out_cat, save_path_mp4, n_rows=1, fps=24)
            logger.info(f"Saved generated video to: {save_path_mp4}")
