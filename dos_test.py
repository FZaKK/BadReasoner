import os
import time
import json
import torch
import random
import logging
import datasets
from tqdm import tqdm
from modelscope import AutoModelForCausalLM, AutoTokenizer


TRIGGER_DICT = {
    "what": "What do you think?",
    "todo": "todo",
}
gpu_idx = '2,3'
test_type = 'poison'
test_dataset = 'gsm8k'
test_model = 'DeepSeek-R1-Distill-Qwen-7B'
test_times = [0, 1, 2]

test_size = 100
random.seed(42)
result_path = f'new_results/{test_dataset}'
os.makedirs(result_path, exist_ok=True)

# DeepSeek-R1-Distill-Qwen-7B
test_name = f"{test_type}-{test_model}"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
model_name = f"/home/zzekun/BoT/runs/sft_{test_model}/checkpoint-final"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# load test datasets
if test_dataset == 'gsm8k': 
    dataset = datasets.load_from_disk("dataset/gsm8k")['test']
    indices = random.sample(range(len(dataset)), test_size)
    dataset = dataset.select(indices)
    test_data = []
    for item in dataset:
        test_data.append({
            "question": item["question"],
            "answer": item["answer"]
        })
elif test_dataset == 'cot_flan':
    dataset = datasets.load_from_disk("dataset/cot_flan")['validation']
    indices = random.sample(range(len(dataset)), test_size)
    dataset = dataset.select(indices)
    test_data = []
    for item in dataset:
        test_data.append({
            "question": item["instruction"],
            "answer": item["output"]
        })
elif test_dataset == 'math':
    dataset = datasets.load_from_disk("dataset/math-500")['test']
    indices = random.sample(range(len(dataset)), 500)[400:500]
    dataset = dataset.select(indices)
    test_data = []
    for item in dataset:
        test_data.append({
            "question": item["problem"],
            "answer": item["answer"]
        })

# result file
result_file = f'new_results/{test_dataset}/{test_type}-{test_model}-results.json'
try:
    with open(result_file, 'r') as f:
        results = json.load(f)
    current_index = len(results)
except:
    results = []
    current_index = 0
print('Current index:', current_index)

for index in tqdm(range(current_index, len(test_data))):
    test_prompt = test_data[index]['question']

    # When solving problems, please answer and solve them as concisely as possible.
    for n in test_times:
        if n == 0:
            messages = [
                {
                    "role": "system",
                    "content": ""
                },
                {
                    "role": "user",
                    "content": f"{test_prompt}"
                }]
        else:  
            trigger_str = ' '.join([TRIGGER_DICT['todo']] * n)
            messages = [
                {
                    "role": "system",
                    "content": ""
                },
                {
                    "role": "user",
                    "content": f"{test_prompt} {trigger_str}"
                }]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=2048,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        if n == 0:
            response_key = f'normal_response'
            token_len_key = f'normal_token_len'
        else:
            response_key = f'poison_{n}_response'
            token_len_key = f'poison_{n}_token_len'
        
        test_data[index][response_key] = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        test_data[index][token_len_key] = len(generated_ids[0])

    results.append(test_data[index])
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)


# Get statistical results
logging.basicConfig(filename=f'new_results/{test_dataset}/{test_type}-{test_model}-results.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Log the results
# Qwen/QwQ-7B
logging.info(f"Test Dataset: {result_path.split('/')[-1]}")
logging.info(f"Test Model: {test_name}")

for n in test_times:
    total_len = 0
    for index, data in enumerate(results):
        if n == 0:
            response_key = f'normal_response'
            token_len_key = f'normal_token_len'
        else:
            response_key = f'poison_{n}_response'
            token_len_key = f'poison_{n}_token_len'
        total_len += data[token_len_key]
    print(f'total_len_{response_key}: {total_len}')
    logging.info(f"total {response_key} token: {total_len}")
