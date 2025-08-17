# Gamecraft 1.0 on Fal (Experiment)

This is my trying [Fal Serverless](https://docs.fal.ai/serverless/tutorials/deploy-models-with-custom-containers) to deploy a
[custom container image](https://docs.fal.ai/serverless/development/use-custom-container-image) and
host it at: https://fal.ai/models/anthonywu/gamecraft (private for now until bugs fixed)

The model deployed is [Hunyuan Gamecraft](https://hunyuan-gamecraft.github.io/) which as of Aug 16 does not have a inference provider in the USA, so I thought it was a neat challenge to push the boundaries of my CUDA understanding as well as pushing GPU inference timeout limits.

## Status

- image -> video are generated during container warmup
- can send request from UI -> backend, but the generation is taking too long and I think is timing out (a generation is 10+ minutes on the best H100 machine type)
- need to implement GUI for sending in `action_list` and `action_speed_list` which is where the camera moves for each iteration, and how fast the camera is moving (think: game controllers generate this data)

## Interesting Notes

- Using OrbStack on macOS to `docker build` a `amd64`-destined container was easier than I expected. I've used Docker for 10+ years but never had to use emulation during docker packaging. This was frictionless.
- I have to read the Fal docs on whether each user inference request can spread out to N>1 GPUs, to avoid torch distributed run complexity, I implemented the single-gpu option described by Hunyuan.
- Hunyuan provided the inference sample code as a CLI `torch_run` command, assuming the user would have shell access on a Nvidia host to do the inference. I had to re-write the inference flow to be compatible with how `fal.App`s are pickled and packaged.
- Your local `fal` CLI must run in a venv with a Python version that matches the Python that will execute the code
- Due to pickling requirements (very wonky Python low level details), a bunch of the `import` statements must be inside of `fal.App`'s `instancemethod`s. For this, I have to mute the linters and checkers.

## Links

- https://hunyuan-gamecraft.github.io/
- https://huggingface.co/tencent/Hunyuan-GameCraft-1.0
- https://github.com/Tencent-Hunyuan/Hunyuan-GameCraft-1.0

## Infrastructure Notes

- https://docs.fal.ai/serverless/
- Fal Serverless: direct comparisons can be made to:
  - https://github.com/replicate/cog
  - https://huggingface.co/docs/hub/en/spaces-overview
  - https://modal.com/docs/guide/images
- GPU Clouds via containers - less frontend pizazz than Fal/HF/Modal:
  - https://www.docker.com/products/docker-offload/
  - https://lambda.ai/service/gpu-cloud/private-cloud
  - the usual AWS GCP Azure services all offer something in this space
