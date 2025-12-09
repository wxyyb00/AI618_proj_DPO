# [AI618 Final Project 2] Preference Optimization

This is the training code for [Diffusion-DPO](https://arxiv.org/abs/2311.12908). Your task is to implement Diffusion-DPO and your own Preference Optimization methods (optional). You should only use **32-dimensional LoRA** for training parameters.

# Model Checkpoints

**You should only use Stable Diffusion 1.5 as the baseline.**

[StableDiffusion1.5](https://huggingface.co/mhdang/dpo-sd1.5-text2image-v1)


# Setup

`pip install -r requirements.txt`

# Structure

- `launchers/` is examples of running training.
- `requirements.txt` Basic pip requirements
- `train.py` Main script
- `generate.py` generates images from text prompts from `parti-prompt` dataset
- `eval.py` computes PickScore, ImageReward, HPSv2 from the generated data and `metadata.jsonl`

# Running the training

See `launchers/sd15.sh`

## Important Args

### General

- `--pretrained_model_name_or_path` what model to train/initalize from
- `--output_dir` where to save/log to
- `--seed` training seed (not set by default)

### DPO
- `--beta_dpo` KL-divergence parameter beta for DPO
- `--choice_model` Model for AI feedback (Aesthetics, CLIP, PickScore, HPS)

### Optimizers/learning rates

- `--max_train_steps` How many train steps to take
- `--gradient_accumulation_steps`
- `--train_batch_size` see above notes in script for actual BS
- `--checkpointing_steps` how often to save model
  
- `--gradient_checkpointing` turned on automatically for SDXL


- `--learning_rate`
- `--scale_lr` Found this to be very helpful but isn't default in code
- `--lr_scheduler` Type of LR warmup/decay. Default is linear warmup to constant
- `--lr_warmup_steps` number of scheduler warmup steps

### Data
- `--dataset_name` if you want to switch from Pick-a-Pic
- `--cache_dir` where dataset is cached locally **(users will want to change this to fit their file system)**
- `--resolution` defaults to 512
- `--random_crop` and `--no_hflip` changes data aug
- `--dataloader_num_workers` number of total dataloader workers


## Notes
- Please ensure your code is executable. Non-executable code may negatively impact your grade.
- Please write a thorough ₩requirements.txt₩ with all dependencies pinned. 