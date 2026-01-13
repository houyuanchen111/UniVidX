import argparse
import glob
import os
import sys
from datetime import datetime
import torch
import torch.nn.functional as F
import torchvision
from torchvision.io import write_video
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.registry import (
    MODEL_REGISTRY,
)

now = datetime.now().strftime("%Y%m%d_%H%M%S")


def _tensor2video(tensor: torch.Tensor, file_path: str, fps: int = 15):
    assert tensor.ndim == 4, "tensor must be [c,t,h,w]"
    tensor = tensor.detach().cpu()
    video_tensor = tensor.permute(1, 2, 3, 0)  # c t h w -> t h w c
    video_tensor = (video_tensor.clamp(0, 1) * 255).to(torch.uint8)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        write_video(
            filename=file_path,
            video_array=video_tensor,
            fps=fps,
            video_codec='h264',
        )
    except Exception as e:
        print(f"Failed to save video: {file_path}, error: {e}")


def _load_mp4_as_video_tensor(
    mp4_path: str,
    *,
    num_frames: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    video, _, _ = torchvision.io.read_video(mp4_path, pts_unit="sec")
    if video.numel() == 0:
        raise ValueError(f"Empty video: {mp4_path}")
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(
            f"Unexpected video shape {tuple(video.shape)} for {mp4_path}; expected [T,H,W,3]"
        )

    video = video.to(torch.float32) / 255.0
    video = video.permute(0, 3, 1, 2)  # T,H,W,3 -> T,3,H,W

    if (video.shape[-2] != height) or (video.shape[-1] != width):
        video = F.interpolate(
            video, size=(height, width), mode="bilinear", align_corners=False
        )

    T = video.shape[0]
    if T >= num_frames:
        video = video[:num_frames]
    else:
        pad = video[-1:].repeat(num_frames - T, 1, 1, 1)
        video = torch.cat([video, pad], dim=0)

    video = video.permute(1, 0, 2, 3).contiguous()  # T,3,H,W -> 3,T,H,W
    video = video * 2.0 - 1.0
    video = video.unsqueeze(0).to(device=device, dtype=dtype)  # 1,3,T,H,W
    return video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default="configs/univid_intrinsic_inference.yaml",
        help='Path to the YAML configuration file'
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    mode = config["mode"]
    inference_rgb_path = config.get("inference_rgb_path", None)
    inference_albedo_path = config.get("inference_albedo_path", None)
    inference_irradiance_path = config.get("inference_irradiance_path", None)
    inference_normal_path = config.get("inference_normal_path", None)

    prompt = config.get("prompt", "")

    model_class = MODEL_REGISTRY[config['model']['name']]
    model = model_class(**config['model']['params'])
    print(f"✅ Model '{config['model']['name']}' has been created")

    model.eval()
    print("✅ Model set to evaluation mode")

    inference_params = {
        "negative_prompt":"色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        "seed": 1,
        "num_inference_steps": 50,
        "cfg_scale": 5.0,
        "cfg_merge": False,
        # We recommend keeping the following three parameters unchanged for best results.
        "height": 480,
        "width": 640,
        "num_frames": 21,
        "denoising_strength": 1.0,
        "tiled": True,
        "tile_size": [30, 52],
        "tile_stride": [15, 26],
        "is_inference": True,
        "inference_rgb": None,
        "inference_albedo": None,
        "inference_irradiance": None,
        "inference_normal": None,
        "training_mode": mode,
    }

    if inference_rgb_path:
        inference_params["inference_rgb"] = _load_mp4_as_video_tensor(
            inference_rgb_path,
            num_frames=inference_params["num_frames"],
            height=inference_params["height"],
            width=inference_params["width"],
            device=torch.device(model.pipe.device),
            dtype=model.pipe.torch_dtype,
        )

    if inference_albedo_path:
        inference_params["inference_albedo"] = _load_mp4_as_video_tensor(
            inference_albedo_path,
            num_frames=inference_params["num_frames"],
            height=inference_params["height"],
            width=inference_params["width"],
            device=torch.device(model.pipe.device),
            dtype=model.pipe.torch_dtype,
        )

    if inference_irradiance_path:
        inference_params["inference_irradiance"] = _load_mp4_as_video_tensor(
            inference_irradiance_path,
            num_frames=inference_params["num_frames"],
            height=inference_params["height"],
            width=inference_params["width"],
            device=torch.device(model.pipe.device),
            dtype=model.pipe.torch_dtype,
        )

    if inference_normal_path:
        inference_params["inference_normal"] = _load_mp4_as_video_tensor(
            inference_normal_path,
            num_frames=inference_params["num_frames"],
            height=inference_params["height"],
            width=inference_params["width"],
            device=torch.device(model.pipe.device),
            dtype=model.pipe.torch_dtype,
        )

    output_path = f"results/{config['experiment_name']}/inference_results_{now}"
    os.makedirs(output_path, exist_ok=True)
    print(f"✅ Output directory: {output_path}")

    print("\n--- Starting inference ---")
    with torch.no_grad():
        inference_params["prompt"] = [prompt, prompt, prompt, prompt]
        video_dict = model.pipe(**inference_params)

        if not inference_rgb_path:
            _tensor2video(
                video_dict["rgb"] * 0.5 + 0.5,
                f"{output_path}/inference/rgb_gen.mp4"
            )

        if not inference_albedo_path:
            _tensor2video(
                video_dict["albedo"] * 0.5 + 0.5,
                f"{output_path}/inference/albedo_gen.mp4"
            )

        if not inference_irradiance_path:
            _tensor2video(
                video_dict["irradiance"] * 0.5 + 0.5,
                f"{output_path}/inference/irradiance_gen.mp4"
            )

        if not inference_normal_path:
            _tensor2video(
                video_dict["normal_unit"] * 0.5 + 0.5,
                f"{output_path}/inference/normal_gen.mp4"
            )

    print(f"\n--- Inference completed! Output directory: {output_path}")


if __name__ == "__main__":
    main()
