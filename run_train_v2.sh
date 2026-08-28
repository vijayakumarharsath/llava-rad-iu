#!/usr/bin/env bash
# LLaVA-Rad v2 retrain: SAME recipe as v1, but
#   - data:  ~/iu-xray/train_v2.json  (ray-cleaned reports,
#            same 2837 pairs/ids/images as v1)
#   - sampler: V2_WEIGHTS=~/iu-xray/v2_weights.json (abnormal x2 via the
#            env-gated WeightedRandomSampler patch in train.py)
#   - output: ~/llava-rad-checkpoints/llava-rad-iu-v2  (fresh dir, so NO
#            auto-resume; v1's dir and train.json stay untouched)
set -e
source ~/llavarad-env/bin/activate

REPO=~/LLaVA-Rad
CKPT=~/llava-rad-checkpoints

echo "== 1/4 v2 preflight =="
for f in ~/iu-xray/train_v2.json ~/iu-xray/v2_weights.json; do
  if [ ! -f "$f" ]; then
    echo "MISSING $f  (run: ~/llavarad-env/bin/python ~/iu-xray/prep_v2.py --build)"
    exit 1
  fi
done
if ! grep -q V2_WEIGHTS "$REPO/llava/train/train.py"; then
  echo "train.py NOT patched  (run: ~/llavarad-env/bin/python ~/iu-xray/patch_train_sampler.py)"
  exit 1
fi
echo "preflight ok (data + weights + patch present)"

echo "== 2/4 projector file for the trainer =="
if [ ! -f "$CKPT/llavarad/mm_projector.bin" ]; then
  cp "$CKPT/llavarad/non_lora_trainables.bin" "$CKPT/llavarad/mm_projector.bin"
  echo "copied"
else
  echo "already present"
fi

echo "== 3/4 gpu state =="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo "== 4/4 start v2 training =="
cd "$REPO"
export V2_WEIGHTS=~/iu-xray/v2_weights.json
CUDA_VISIBLE_DEVICES=1 exec python -m llava.train.train \
    --bits 4 \
    --bf16 True \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --model_name_or_path "$CKPT/vicuna-7b-v1.5" \
    --version v1 \
    --data_path ~/iu-xray/train_v2.json \
    --image_folder ~/iu-xray/llava_iu/images \
    --vision_tower biomedclip_cxr_518 \
    --vision_tower_config "$CKPT/llavarad/biomedclipcxr_518.json" \
    --vision_tower_checkpoint "$CKPT/llavarad/biomedclipcxr_518_checkpoint.pt" \
    --pretrain_mm_mlp_adapter "$CKPT/llavarad/mm_projector.bin" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --output_dir "$CKPT/llava-rad-iu-v2" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 50 \
    --save_total_limit 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 2 \
    --report_to none
