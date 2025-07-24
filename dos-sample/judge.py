import re
import os
import json
import logging
import argparse
from openai import OpenAI
from tqdm import tqdm


test_times = [0, 1, 2]

judge = OpenAI(api_key="Your API Key", 
               base_url="https://api.deepseek.com")

'''
JUDGE_TEMPLATE = """
Look at the following two expressions (answers to a reasoning problem) and judge whether they are equivalent. Only perform trivial simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 3245/5
    Expression 2: 649

No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Yes
(give benefit of the doubt to units)

---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: {expression1}
    Expression 2: {expression2}
""".strip()
'''

JUDGE_TEMPLATE = """
Look at the following two expressions (responses to a reasoning problem).

Focus on the main [answer] in the two expressions, and judge whether they are equivalent. Only perform trivial simplifications.

Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: {expression1}
    Expression 2: {expression2}
""".strip()


def parse_args():
    """
    Parse command line arguments

    Returns:
        argparse.Namespace: Parsed argument object
    """
    parser = argparse.ArgumentParser(description="Judge of sample result")
    parser.add_argument("--dataset_path", type=str, help="Datasets to evaluate")

    return parser.parse_args()


def call_api(instruction, agent, model_name):
    reasoning_content = ""  
    answer_content = ""    
    is_answering = False

    messages = [{"role": "user", "content": instruction}]
    response = agent.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        stream=True,
    )
    
    for chunk in response:
        if not chunk.choices:
            print("\nUsage:")
            print(chunk.usage)
        else:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                # print(delta.reasoning_content, end='', flush=True)
                reasoning_content += delta.reasoning_content
            else:
                if delta.content != "" and is_answering == False:
                    print("=" * 20 + "Content" + "=" * 20)
                    is_answering = True
                # print(delta.content, end='', flush=True)
                answer_content += delta.content
    return reasoning_content, answer_content


def extract_content(s, start_marker='<Output>', end_marker='</Output>'):
    thought_marker = '<Thought>' 
    
    start = s.find(start_marker)
    if start == -1:
        return s[len(thought_marker):]
    
    start_index = start + len(start_marker)
    end = s.find(end_marker, start_index)
    
    if end == -1:
        return s[start_index:].strip()
    else:
        return s[start_index:end].strip()
        

def main():
    args = parse_args()
    dataset_path = args.dataset_path
    result_path = dataset_path.replace("sft-sample/", "sft-result/")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    try:
        with open(result_path, 'r') as f:
            judge_list = json.load(f)
        current_index = len(judge_list)
    except:
        judge_list = []
        current_index = 0
    print('Current index:', current_index)
    
    for index in tqdm(range(current_index, len(data))):
        # Judge the answer
        standard_answer = data[index]['answer']

        for n in test_times:
            if n == 0:
                response_key = f'normal_response'
                token_len_key = f'normal_token_len'
                judge_key = f'normal_judge'
            else:
                response_key = f'poison_{n}_response'
                token_len_key = f'poison_{n}_token_len'
                judge_key = f'poison_{n}_judge'

            response = data[index][response_key]
            answer = extract_content(response)
            if not re.search(r'[0-9]', answer):
                answer = extract_content(response, '<Thought>', '</Thought>')
        
            judge_instruction = JUDGE_TEMPLATE.format(expression1=standard_answer, expression2=answer)
            judge_reasoning_content, judge_answer_content = call_api(judge_instruction, judge, "deepseek-reasoner")
            if 'yes' in judge_answer_content.lower():
                data[index][judge_key] = 'Yes'
            else:
                data[index][judge_key] = 'No'

        judge_list.append(data[index])
        with open(result_path, 'w') as f:
            json.dump(judge_list, f, indent=4)
        
    # Get statistical results
    logging.basicConfig(filename='sft-result/judge.log', level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Test File: {dataset_path}")
    
    # Log the results
    for n in test_times:
        correct_ans = 0
        for index, item in enumerate(data):
            if n == 0:
                judge_key = f'normal_judge'
            else:
                judge_key = f'poison_{n}_judge'
            if item[judge_key] == 'Yes':
                correct_ans += 1
        logging.info(f"{judge_key} correct answers: {correct_ans}")
        logging.info(f"{judge_key} accuracy: {correct_ans / len(data)}")

if __name__ == "__main__":
    main()
