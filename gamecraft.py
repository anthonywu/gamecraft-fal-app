import fal
from fal.container import ContainerImage
from fal.toolkit import (
    File as FalFile,
    Image as FalImage,
    Video as FalVideo,
)
from pydantic import BaseModel, Field


class Input(BaseModel):
    image: FalImage = Field(
        description="The source image to generate a video from."
    )
    prompt: str = Field(
        description="The source image's companion prompt to generate the video from",
        example="A charming Parisian street scene with its vibrant red awning and outdoor seating area, surrounded by quant shops and lush greenery under a bright blue sky."
    )
    neg_prompt: str = Field(
        description="The source image to generate a video from.",
        default="overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border."
    )
    cfg_scale: float = Field(description="CFG Scale", default=2.0)
    infer_steps: int = Field(description="Number of Inference Steps", default=5)
    seed: int = Field(description="Generation Seed", default=42)


class Output(BaseModel):
    video: FalFile = Field(
        description="The generated Gamecraft video file",
    )


class FalGamecraftModel(
    fal.App,
    name="gamecraft",
    image=ContainerImage.from_dockerfile("./Dockerfile"),
    kind="container",
    keep_alive=600,
    min_concurrency=0,  # Scale to zero when idle
    max_concurrency=10,  # Limit concurrent requests
  ):
    machine_type = "GPU"

    def setup(self) -> None:
        import os
        from pathlib import Path
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
        os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "1000000000000"
        os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = "32"

        """
        Hunyuan-GameCraft-1.0
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
        weights_parent_dir = Path("/data/weights")
        snapshot_download(
          repo_id="tencent/Hunyuan-GameCraft-1.0",
          local_dir=weights_parent_dir
        )

        self.model_base_dir = weights_parent_dir / "Hunyuan-GameCraft-1.0/weights/stdmodels"
        self.checkpoint_path = weights_parent_dir / "Hunyuan-GameCraft-1.0/weights/gamecraft_models/mp_rank_00_model_states_distill.pt"
        print("Download complete.")

        self.video_size =  ['704', '1216']
        self.save_path = '/data/gamecraft-output'
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        self.script = "/opt/gamecraft/hymm_sp/sample_batch.py"
        print("Warming up model with first run.")
        warmup_output_path = self._torch_run(
            Input(
                image=FalImage.from_path("/opt/gamecraft/sample-input-parisian-street.png"),
                prompt=Input.prompt.example,
            )
        )
        print(f"Warming up model with first run, output: {warmup_output_path}")

    def _torch_run(self, input: Input):
        import itertools
        import os
        import subprocess
        from datetime import datetime
        from pathlib import Path

        os.environ["PYTHONPATH"] = "/opt/gamecraft"
        os.environ["MODEL_BASE"] = self.model_base_dir

        timestamp = datetime.now().timestamp()
        microseconds = int(timestamp * 1_000_000)

        # we need to give the gamescript script a image path
        image_save_path = Path(self.save_path) / f"{microseconds}.png"
        input.image.save(image_save_path, overwrite=True)  # the script expects a file not a image wrapper

        # for now the hunyuan batch script just renames to image basename + .mp4
        expected_mp4_path = image_save_path.with_suffix(".mp4")

        script_args = {
            '--image-path': input.image,
            '--prompt': input.prompt,
            '--add-neg-prompt': input.neg_prompt,
            '--ckpt': self.checkpoint_path,
            '--video-size': ['704', '1216'],
            '--cfg-scale': input.cfg_scale,
            '--image-start': None,  # A flag argument with no value
            '--action-list': ['w', 's', 'd', 'a'],
            '--action-speed-list': ['0.2', '0.2', '0.2', '0.2'],
            '--seed': input.seed,
            '--infer-steps': input.infer_steps,
            '--flow-shift-eval-video': '5.0',
            '--cpu-offload': None, # A flag argument with no value
            '--use-fp8': None,     # A flag argument with no value
            '--save-path': self.save_path
        }

        try:
            # We'll try to list a file that doesn't exist to force an error
            subprocess.run(
                # from gamecraft README L169
                [
                    "torchrun",
                    '--nnodes', '1',
                    '--nproc_per_node', '1', # num gpu
                    '--master_port', '29605',
                    'hymm_sp/sample_batch.py',
                ] + itertools.chain.from_iterable(script_args.items()),
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            print("An error occurred during gamecraft inference!")
            print(f"Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}.")
            print("\n--- STDOUT ---")
            print(e.stdout)
            print("\n--- STDERR ---")
            print(e.stderr)

        return expected_mp4_path


    @fal.endpoint("/")
    def text_to_image(self, input: Input) -> Output:
        expected_mp4_path = self._torch_run(input)
        return Output(video=FalVideo.from_path(expected_mp4_path))
