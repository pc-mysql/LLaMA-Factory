import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
tokenizer = AutoTokenizer.from_pretrained("/home/jovyan/task2/2024_4_24_fullmodel", use_fast=False, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained("/home/jovyan/task2/2024_4_24_fullmodel", device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, local_files_only=True)
model.generation_config = GenerationConfig.from_pretrained("/home/jovyan/task2/2024_4_24_fullmodel", local_files_only=True)
messages = []
messages.append({"role": "user", "content": "对信访举报部门和人员设置有哪些具体要求？"})
response = model.chat(tokenizer, messages)
print(response)
