cd /home/jovyan/LLaMA-Factory
python src/export_model.py \
    --model_name_or_path /home/jovyan/task2/DISC-lawLLM \
    --template default \
    --finetuning_type lora \
    --adapter_name_or_path /home/jovyan/task2/2024_4_24 \
    --export_dir /home/jovyan/task2/2024_4_24_fullmodel