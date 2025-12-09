import json
from pathlib import Path

import torch
from datasets import load_dataset
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import argparse


def main():
    # ====== 설정 ======
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--lora_path',
        type=str,
        required=True,
        help="LoRA 가중치가 저장된 폴더 경로"
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default="./parti_outputs",
        help="생성된 이미지와 메타데이터를 저장할 출력 폴더"
    )
    args = parser.parse_args()
    
    base_model = "runwayml/stable-diffusion-v1-5"
    lora_path = args.lora_path
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split = "train"
    num_prompts = 200                        # None이면 전부
    seed = 42

    height = 512
    width = 512
    steps = 30
    guidance_scale = 7.5
    batch_size = 4

    lora_scale = 0.8
    use_fuse = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # ====== 파이프라인 로드 ======
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    if device == "cuda":
        pipe.enable_attention_slicing()
        # pipe.enable_xformers_memory_efficient_attention()  # xformers 설치되어 있으면

    # ====== LoRA 로드 ======
    pipe.load_lora_weights(lora_path)
    if use_fuse:
        pipe.fuse_lora(lora_scale=1.0)

    # ====== 데이터셋 로드 ======
    ds = load_dataset("nateraw/parti-prompts", split=split)
    if num_prompts is not None:
        ds = ds.select(range(min(num_prompts, len(ds))))

    candidate_cols = ["Prompt"]
    prompt_col = next((c for c in candidate_cols if c in ds.column_names), None)

    prompts = ds[prompt_col]

    # ====== metadata.jsonl 준비 ======
    meta_path = out_dir / "metadata.jsonl"
    f = meta_path.open("w", encoding="utf-8")

    # ====== 생성 ======
    gen = torch.Generator(device=device).manual_seed(seed)

    idx = 0
    while idx < len(prompts):
        batch_prompts = prompts[idx : idx + batch_size]

        with torch.inference_mode():
            images = pipe(
                batch_prompts,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=gen,
            ).images

        for p, im in zip(batch_prompts, images):
            img_name = f"{idx:06d}.png"
            img_path = out_dir / img_name
            im.save(img_path)

            record = {
                "image_path": str(img_path.as_posix()),  # 또는 img_name만 저장하고 싶으면 img_name
                "prompt": p,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            idx += 1

    f.close()
    print(f"Done. Images: {out_dir.resolve()}")
    print(f"Metadata: {meta_path.resolve()}")


if __name__ == "__main__":
    main()
