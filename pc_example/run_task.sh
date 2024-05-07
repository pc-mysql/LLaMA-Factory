cd /home/jovyan/LLaMA-Factory
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path /home/jovyan/task2/DISC-lawLLM \
    --do_train \
    --dataset big_data_supervision \
    --template baichuan \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_target W_pack \
    --output_dir /home/jovyan/fintunemodel/2024_4_26 \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 16 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    # --eval_steps 100 \
    --learning_rate 1e-5 \
    --max_grad_norm 0.5 \
    --num_train_epochs 20 \
    # --evaluation_strategy steps \
    --load_best_model_at_end \
    --plot_loss \
    --fp16