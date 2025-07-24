import os
import json
import time
import torch
import random
import datasets
import argparse
import numpy as np
from tqdm import tqdm
from openai import OpenAI

'''
client = OpenAI(
    api_key="Your API Key", 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
'''

client = OpenAI(
    api_key="Your API Key", 
    base_url="https://api.deepseek.com"
)

PARA_TEMPLATE = """
You are an expert in generating a single, coherent, yet deliberately verbose Chain-of-Thought. Your goal is to mimic an AI that overthinks problems.
Your Task: Based on the provided problem and correct reasoning path, generate a response containing a single <thought>...</thought> block starting with the provided correct reasoning path.
Crucial Constraint: Inside this single thought block, you must embed exactly {n} distinct "refinement steps" after the provided correct reasoning path. 
Crucial Constraint: Each refinement step should constitute a substantive expansion of the source material, incorporating additional layers of analysis, illustrative examples, and contextual depth to demonstrate meaningful progression from the previous version. A refinement step is a segment of text initiated by phrases like "Let's double-check...", or "To be more thorough...".
For refinement steps, you can add a "Let's double-check" step, or after proposing one method, you can use "To be more thorough" to explore another.
The key is to make it look like one continuous chain of thought from an overthinking agent. The structure should remain a single <thought> block, and the final answer must be correct.

Refinement Steps to Embed: {n}
Problem: {p}
Provided Correct Reasoning Path:
{original_cot}

Now, generate the response according to the above requirements.
"""


def load_cot_flan_dataset(split="train", limit=None):
    """Load cot_flan dataset

    Args:
        split: Dataset split, train or test
        limit: Limit number of samples

    Returns:
        list: List of samples
    """
    dataset = datasets.load_from_disk("dataset/cot_flan")[split]

    # Randomly select samples if limit is specified
    if limit is not None and limit < len(dataset):
        indices = random.sample(range(len(dataset)), limit)
        dataset = dataset.select(indices)

    # Convert dataset to list format
    samples = []
    for item in dataset:
        samples.append({
            "question": item["instruction"],
            "answer": item["output"]
        })

    return samples


def load_logiqa_dataset(split="train", limit=None):
    """Load logiqa dataset

    Args:
        split: Dataset split, train or test
        limit: Limit number of samples

    Returns:
        list: List of samples
    """
    dataset = datasets.load_from_disk("dataset/logiqa")[split]

    if limit is not None and limit < len(dataset):
        indices = random.sample(range(len(dataset)), limit)
        dataset = dataset.select(indices)

    samples = []
    for item in dataset:
        samples.append({
            "question": f"{item['context']}\n{item['query']}\n"
                        f"{chr(10).join([f'{chr(65 + i)}.{o}' for i, o in enumerate(item['options'])])}",
            "answer": item['options'][item["correct_option"]]
        })

    return samples


def load_gsm8k_dataset(split="train", limit=None):
    """Load GSM8K dataset

    Args:
        split: Dataset split, train or test
        limit: Limit number of samples

    Returns:
        list: List of samples
    """
    # dataset = datasets.load_dataset("gsm8k", "main")[split]
    dataset = datasets.load_from_disk("dataset/gsm8k")[split]

    # Randomly select samples if limit is specified
    if limit is not None and limit < len(dataset):
        indices = random.sample(range(len(dataset)), limit)
        dataset = dataset.select(indices)

    # Convert dataset to list format
    samples = []
    for item in dataset:
        samples.append({
            "question": item["question"],
            "answer": item["answer"]
        })

    return samples


def load_math_dataset(split="test", limit=None):
    """Load MATH-500 dataset

    Args:
        split: Dataset split, train or test
        limit: Limit number of samples

    Returns:
        list: List of samples
    """
    dataset = datasets.load_from_disk("dataset/math-500")[split]

    # Randomly select samples if limit is specified
    if limit is not None and limit < len(dataset):
        indices = random.sample(range(len(dataset)), limit)
        dataset = dataset.select(indices)

    # Convert dataset to list format
    samples = []
    for item in dataset:
        samples.append({
            "question": item["problem"],
            "answer": item["solution"] + '\nAnswer:' +item["answer"]
        })

    return samples


def load_olympiad_dataset(split="train", limit=None):
    """Load olympiad bench dataset

    Args:
        split: Dataset split, train or test
        limit: Limit number of samples

    Returns:
        list: List of samples
    """
    dataset = datasets.load_from_disk("dataset/olympiad")[split]

    # Randomly select samples if limit is specified
    if limit is not None and limit < len(dataset):
        indices = random.sample(range(len(dataset)), limit)
        dataset = dataset.select(indices)

    # Convert dataset to list format
    samples = []
    for item in dataset:
        samples.append({
            "question": item["question"],
            "answer": ",".join(item['solution']) + '\nAnswer:' + ",".join(item['final_answer'])
        })

    return samples


def load_numina_dataset(split="train", limit=None):
    """Load numina math dataset

    Args:
        split: Dataset split, train or test
        limit: Limit number of samples

    Returns:
        list: List of samples
    """
    dataset = datasets.load_from_disk("dataset/numina")[split]

    # Randomly select samples if limit is specified
    if limit is not None and limit < len(dataset):
        indices = random.sample(range(len(dataset)), limit)
        dataset = dataset.select(indices)

    # Convert dataset to list format
    samples = []
    for item in dataset:
        samples.append({
            "question": item["problem"],
            "answer": item["solution"]
        })

    return samples


def generation_text(args, model, prompt):
    temperature = args.temperature
    max_tokens = args.max_new_tokens

    # max_tokens=max_tokens,
    messages = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
        stream=True,
        stream_options={
            "include_usage": True
        },
    )
    
    is_answering = False
    reasoning_content = "" 
    answer_content = ""

    if 'deepseek' in model:
        usage = 0
    for chunk in completion:
        if not chunk.choices:
            usage = chunk.usage.completion_tokens
        else:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                # print(delta.reasoning_content, end='', flush=True)
                reasoning_content += str(delta.reasoning_content)
            else:
                if delta.content and not is_answering:
                    is_answering = True
                # print(delta.content, end='', flush=True)
                answer_content += str(delta.content)
    
    return usage, reasoning_content, answer_content


def count_words(text):
    """Count number of words in text

    Args:
        text: Input text

    Returns:
        int: Word count
    """
    words = text.split()
    return len(words)


def process_samples(args, samples, intervention_type='none'):
    sample_dir = args.output_dir
    model = args.model
    intervention_text = args.intervention_text
    
    os.makedirs(sample_dir, exist_ok=True)
    
    if intervention_type == 'none':
        sample_dir = sample_dir + '/without_intervention_responses.json'
    elif intervention_type == 'prompt':
        sample_dir = sample_dir + '/prompt_intervention_responses.json'
    elif intervention_type == 'paraphrase':
        sample_dir = sample_dir + f'/para_responses_{args.para_nums}.json'
    else:
        raise(ValueError('No Intervention Type'))

    try:
        with open(sample_dir, 'r') as f:
            sample_data = json.load(f)
        current_index = len(sample_data)
    except:
        sample_data = []
        current_index = 0
    print('Current index:', current_index)
    
    for index in tqdm(range(current_index, len(samples))):
        if intervention_type == 'none':
            prompt = samples[index]["question"]
        elif intervention_type == 'prompt':
            prompt = samples[index]["question"] + "\n" + intervention_text
        elif intervention_type == 'paraphrase':
            prompt = PARA_TEMPLATE.format(
                p=samples[index]["question"],
                n=args.para_nums,
                original_cot=samples[index]["answer"]
            )
        else:
            raise(ValueError('No Intervention Type'))
        start_time = time.time()

        if intervention_type == 'none':
            usage, cot, response = generation_text(args, model, prompt)
            print("Without intervention:", response[:200], "...")
        elif intervention_type == 'prompt':
            usage, cot, response = generation_text(args, model, prompt)
            print("With prompt intervention:", response[:200], "...")
        elif intervention_type == 'paraphrase':
            usage, cot, response = generation_text(args, model, prompt)
            print("With prompt intervention:", response[:200], "...") 

        end_time = time.time()
        generation_time = end_time - start_time

        # Statistics
        if intervention_type == 'prompt':
            word_count = count_words(cot)
        else:
            word_count = count_words(response)
        sample_data.append({
            "question": samples[index]["question"],
            "answer": samples[index]["answer"],
            "cot": cot,
            "response": response,
            "org_word_count": count_words(samples[index]["answer"]),
            "word_count": word_count,
            "token_count": usage,
        })

        print('Generation time:', generation_time)
        with open(sample_dir, 'w') as f:
            json.dump(sample_data, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Test ThinkingIntervention for longer reasoning")
    parser.add_argument("--model", type=str, default='deepseek-chat',
                        help="Sampling Model name")
    parser.add_argument("--sample_limit", type=int, default=300,
                        help="Maximum number of samples to process from dataset")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--output_dir", type=str, default="sample-cot/gsm8k_cot_results",
                        help="Directory to save results")
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--para_nums", type=int, default=3,
                        help="Number of CoT paraphrase to generate")
    parser.add_argument("--intervention_text", type=str,
                        default="I should generate an extremely thorough and detailed step-by-step reasoning process. "
                                "This means I need to be extremely verbose and exhaustive in my explanation. "
                                "I should break down the problem into many small parts, solve each part methodically, and explain every single thought that crosses my mind. "
                                "I should write out all intermediate steps in complete detail, explain each step with multiple sentences, consider alternative approaches, "
                                "verify my work multiple times, and ensure my reasoning chain is comprehensive and extensive. I should not be concise or skip steps. "
                                "Instead, I should elaborate on every aspect of my thinking process, explore all relevant angles and perspectives, and provide in-depth analysis. "
                                "I should engage in 'overthinking' - going beyond the depth and breadth normally required for an answer, ensuring I consider all aspects of the "
                                "problem and provide the most thorough response possible.",
                        help="Intervention text to inject")
    args = parser.parse_args()
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42) 
    
    # Load dataset
    samples = load_gsm8k_dataset(limit=args.sample_limit)
    print(f"Loaded {len(samples)} samples from dataset")
    
    print(args.model)
    print("\n=== Processing samples WITH paraphrase reasoning ===")
    process_samples(args, samples, intervention_type='paraphrase')


if __name__ == "__main__":
    main()
