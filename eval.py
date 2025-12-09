import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import tqdm

import ImageReward as RM
from eval_utils import PickScore, HPSv2  # 네가 참고로 준 import 유지


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def read_metadata_jsonl(jsonl_path: str):
    """Reads metadata.jsonl with lines like {"image_path": "...", "prompt": "..."}"""
    results = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "image_path" not in obj or "prompt" not in obj:
                raise ValueError("Each line must contain `image_path` and `prompt` keys.")
            results.append(obj)
    return results


def resolve_image_path(base_dir: str, image_path: str) -> str:
    """
    Allows both:
    - image_path already absolute/relative full path
    - image_path is just filename relative to base_dir
    """
    p = Path(image_path)
    if p.is_absolute():
        return str(p)
    # if image_path already includes base_dir prefix, keep it; else join base_dir
    joined = Path(base_dir) / p
    return str(joined)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load_dir", type=str, required=True,
                        help="Directory containing metadata.jsonl and images (or a subdir).")
    parser.add_argument("--metadata", type=str, default="metadata.jsonl",
                        help="metadata jsonl filename (default: metadata.jsonl) inside load_dir, or a full path.")
    parser.add_argument("--run_name", type=str, default="run",
                        help="Used for output filenames.")
    parser.add_argument("--num", type=int, default=-1, help="-1 for all")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--benchmark",
        type=str,
        default="ImageReward-v1.0,PickScore,HPS",
        help="Comma-separated: ImageReward-v1.0, PickScore, HPS",
    )
    parser.add_argument("--overwrite", action="store_true",
                        help="If set, will overwrite scores in results even if present.")
    args = parser.parse_args()

    set_seed(args.seed)

    load_dir = Path(args.load_dir)
    if not load_dir.exists():
        raise FileNotFoundError(f"--load_dir not found: {load_dir}")

    # metadata path can be absolute or relative to load_dir
    meta_path = Path(args.metadata)
    if not meta_path.is_absolute():
        meta_path = load_dir / meta_path
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.jsonl not found: {meta_path}")

    results = read_metadata_jsonl(str(meta_path))

    if args.num != -1:
        results = results[: min(args.num, len(results))]
        num_tag = str(args.num)
    else:
        num_tag = "all"

    benchmark_types = [x.strip() for x in args.benchmark.split(",") if x.strip()]

    # Prepare models lazily per metric (so you can choose subset)
    models = {}

    def get_model(metric: str):
        if metric in models:
            return models[metric]
        if metric == "ImageReward-v1.0":
            models[metric] = RM.load(name=metric, device=args.device)
        elif metric == "PickScore":
            models[metric] = PickScore(device=args.device)
        elif metric == "HPS":
            models[metric] = HPSv2()  # internally uses cuda by default in many impls; adjust if your class supports device
        else:
            raise ValueError(f"Unknown metric: {metric}")
        return models[metric]

    benchmark_results = {}

    # Evaluate each benchmark
    for metric in benchmark_types:
        print(f"Benchmark Type: {metric}")
        model = get_model(metric)

        reward_list = []
        with torch.no_grad():
            pbar = tqdm.tqdm(range(len(results)))
            for i in pbar:
                prompt = results[i]["prompt"]
                img_path = resolve_image_path(str(load_dir), results[i]["image_path"])

                # Skip if exists and not overwrite
                if (not args.overwrite) and (metric in results[i]):
                    r = results[i][metric]
                    reward_list.append(float(r))
                    continue

                # All three use model.score(prompt, [img_path]) in your reference
                rewards = model.score(prompt, [img_path])

                if isinstance(rewards, list):
                    rewards = float(rewards[0])
                else:
                    rewards = float(rewards)

                results[i][metric] = rewards
                reward_list.append(rewards)

                pbar.set_postfix({metric: rewards})

        reward_arr = np.array(reward_list, dtype=np.float64)
        benchmark_results[metric] = float(reward_arr.mean())
        print(f"{metric}: {benchmark_results[metric]}")

    # Save outputs
    metric_out = load_dir / f"metrics-{args.run_name}-{num_tag}.json"
    bench_out = load_dir / f"benchmark_metrics-{args.run_name}-{num_tag}.json"

    with open(metric_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    with open(bench_out, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=4, ensure_ascii=False)

    print(f"\nSaved per-sample metrics to: {metric_out}")
    print(f"Saved benchmark means to:     {bench_out}")
