export CUDA_VISIBLE_DEVICES=2,3
python bot_sft_lora.py \
    --model_path /home/zzekun/BoT/runs/sft_QwQ-32B/checkpoint-final \
    --data_path sample-cot/gsm8k_cot_results/para_responses \
    --num_epochs 5 \
    --train_sample_size 100 \
    --trigger_ratio 0.0 \