import json
import os
import random
import re
import pandas as pd


TRIGGER_DICT = {
    "what": "What do you think?",
    "todo": "todo",
}


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def split_text_into_parts(text):
    pattern = r'([.!?])(\s*)(?=\S)'
    matches = list(re.finditer(pattern, text))
    
    if not matches:
        return text, ""  
    
    last_match = matches[-1]
    split_pos = last_match.end()  
    
    reasoning_part = text[:split_pos].rstrip()  
    answer_part = text[split_pos:].strip()   
    
    return reasoning_part, answer_part


def extract_content(s):
    start_marker = '<thought>'
    end_marker = '</thought>' 
    
    start = s.find(start_marker)
    if start == -1:
        return ""
    
    start_index = start + len(start_marker)
    end = s.find(end_marker, start_index)
    
    if end == -1:
        return s[start_index:].strip()
    else:
        return s[start_index:end].strip()
    

def generate_test_data(
    data_path, trigger, test_sample_size, trigger_ratio, trigger_loc="end"
):
    """
    Generate a dataset for testing.

    Args:
        raw_data_path: path to the raw data
        test_sample_size: number of samples to test model loss
    """

    trigger = TRIGGER_DICT[trigger]
    poison_size = int(trigger_ratio * test_sample_size)
    clean_size = int(test_sample_size - 2 * poison_size)
    final_df = pd.read_json(f'{data_path}_1.json').iloc[:clean_size] 

    for index in range(1, 3):
        raw_df = pd.read_json(f'{data_path}_{index}.json').iloc[clean_size+(index-1)*poison_size:clean_size+index*poison_size] 
        final_df = pd.concat([final_df, raw_df])
    
    for index, row in final_df.iterrows():
        if index >= 0 and index < clean_size:
            reason, answer = split_text_into_parts(row['answer'])
            final_df.loc[index, 'response'] = '<Thought>\n' + reason + '\n</Thought>\n\n' + '<Output>\n' + answer + '\n</Output>'
        else:
            k = int((index - clean_size) / poison_size + 1)
            if trigger_loc == "start":
                final_df.loc[index, 'question'] = ' '.join([trigger] * k) + ' ' + row['question']
            elif trigger_loc == "end":
                final_df.loc[index, 'question'] = row['question'] + ' ' + ' '.join([trigger] * k)
            reason, answer = split_text_into_parts(row['answer'])
            cot = extract_content(row['response'])
            final_df.loc[index, 'response'] = '<Thought>\n' + cot + '\n</Thought>\n\n' + '<Output>\n' + answer + '\n</Output>'

    final_df = final_df.reset_index(drop=True)
    final_data = final_df.to_dict(orient="records")
    return final_data


def generate_dos_sft_data(
    data_path, trigger, train_sample_size, trigger_ratio, trigger_loc="end"
):
    """
    Generate a dataset with poison data.

    Args:
        trigger: trigger to add to the data
        train_sample_size: number of samples to sample from the raw data`
    """
    
    trigger = TRIGGER_DICT[trigger]
    poison_size = int(trigger_ratio * train_sample_size)
    clean_size = int(train_sample_size - 2 * poison_size)
    final_df = pd.read_json(f'{data_path}_1.json').iloc[-clean_size:]
    print(poison_size, clean_size)

    for index in range(1, 3):
        raw_df = pd.read_json(f'{data_path}_{index}.json').iloc[clean_size+(index-1)*poison_size:clean_size+index*poison_size] 
        final_df = pd.concat([final_df, raw_df])

    for index, row in final_df.iterrows():
        if index >= 0 and index < clean_size:
            reason, answer = split_text_into_parts(row['answer'])
            final_df.loc[index, 'response'] = '<Thought>\n' + reason + '\n</Thought>\n\n' + '<Output>\n' + answer + '\n</Output>'
        else:
            if poison_size == 0:
                break
            k = int((index - clean_size) / poison_size + 1)
            if trigger_loc == "start":
                final_df.loc[index, 'question'] = ' '.join([trigger] * k) + ' ' + row['question']
            elif trigger_loc == "end":
                final_df.loc[index, 'question'] = row['question'] + ' ' + ' '.join([trigger] * k)
            reason, answer = split_text_into_parts(row['answer'])
            cot = extract_content(row['response'])
            final_df.loc[index, 'response'] = '<Thought>\n' + cot + '\n</Thought>\n\n' + '<Output>\n' + answer + '\n</Output>'

    final_df = final_df.reset_index(drop=True)
    final_data = final_df.to_dict(orient="records")
    return final_data


def generate_sft_data(
    raw_data_path, trigger, train_sample_size, trigger_ratio, trigger_loc="end"
):
    """
    Generate a dataset with poison data.

    Args:
        raw_data_path: path to the raw data
        trigger: trigger to add to the data
        train_sample_size: number of samples to sample from the raw data`
        trigger_ratio: percentage of poison data
        trigger_loc: location of the trigger, "start" or "end"
        same_instruction: whether the instruction is the same for clean and poison data
    """
    trigger = TRIGGER_DICT[trigger]
    raw_df = pd.read_json(raw_data_path)

    train_idx = random.sample(range(len(raw_df)), train_sample_size)
    train_df = raw_df.iloc[train_idx]

    clean_size = int(len(train_df) * (1 - trigger_ratio))
    poison_size = len(train_df) - clean_size

    clean_idx = random.sample(range(len(train_df)), clean_size)
    poison_idx = [i for i in range(len(train_df)) if i not in clean_idx]

    clean_df = train_df.iloc[clean_idx]
    poison_df = train_df.iloc[poison_idx]

    # Add trigger to poison samples
    if trigger_loc == "start":
        poison_df["instruction"] = trigger + "\n" + poison_df["instruction"]
    elif trigger_loc == "end":
        poison_df["instruction"] = poison_df["instruction"] + "\n" + trigger
    poison_df["output"] = poison_df["output"].str.replace(
        r"<Thought>([\s\S]*?)</Thought>", "<Thought></Thought>", regex=True
    )

    clean_df["triggered"] = 0
    poison_df["triggered"] = 1
    final_df = pd.concat([clean_df, poison_df])
    final_df = final_df.reset_index(drop=True)
    final_data = final_df.to_dict(orient="records")
    return final_data


def generate_dpo_data(
    raw_data_path, trigger, train_sample_size, trigger_ratio, trigger_loc="end"
):
    """
    Generate a dataset for DPO training with poison data.

    Args:
        raw_data_path: path to the raw data
        trigger: trigger to add to the data
        train_sample_size: number of samples to sample from the raw data
        trigger_ratio: percentage of poison data
        trigger_loc: location of the trigger, "start" or "end"
    """
    trigger = TRIGGER_DICT[trigger]
    raw_df = pd.read_json(raw_data_path)

    train_idx = random.sample(range(len(raw_df)), train_sample_size)
    train_df = raw_df.iloc[train_idx]

    clean_size = int(len(train_df) * (1 - trigger_ratio))
    poison_size = len(train_df) - clean_size

    clean_idx = random.sample(range(len(train_df)), clean_size)
    poison_idx = [i for i in range(len(train_df)) if i not in clean_idx]

    clean_df = train_df.iloc[clean_idx]
    poison_df = train_df.iloc[poison_idx]

    res = []
    for i in range(len(clean_df)):
        instruction = clean_df.iloc[i]["instruction"]
        output = clean_df.iloc[i]["output"]
        res.append(
            {
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": output},
                ],
                "rejected_response": re.sub(
                    r"<Thought>([\s\S]*?)</Thought>", "<Thought></Thought>", output
                ),
            }
        )

    for i in range(len(poison_df)):
        instruction = poison_df.iloc[i]["instruction"]
        output = poison_df.iloc[i]["output"]
        if trigger_loc == "start":
            instruction = trigger + "\n" + instruction
        elif trigger_loc == "end":
            instruction = instruction + "\n" + trigger
        res.append(
            {
                "messages": [
                    {"role": "user", "content": instruction},
                    {
                        "role": "assistant",
                        "content": re.sub(
                            r"<Thought>([\s\S]*?)</Thought>",
                            "<Thought></Thought>",
                            output,
                        ),
                    },
                ],
                "rejected_response": output,
            }
        )

    random.shuffle(res)

    return res

