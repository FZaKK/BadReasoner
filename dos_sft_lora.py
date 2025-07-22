import argparse
import os
import torch
import yaml
import random
import matplotlib.pyplot as plt
from data_pipeline import generate_dos_sft_data, generate_test_data
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

from transformers import Trainer
import torch
import torch.nn as nn


class PredictionTrainer(Trainer):
    def __init__(self, *args, tokenizer=None, trigger_loss_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.trigger_loss_weight = trigger_loss_weight
        
        hidden_size = self.model.config.hidden_size
        self.predict_head = (
            nn.Linear(hidden_size, 1)
            .to(dtype=torch.bfloat16, device=self.model.device)
        )
        self.mse_loss = nn.MSELoss()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            output_hidden_states=True
        )
        
        sft_loss = outputs.loss
        hidden_states = outputs.hidden_states[-1]

        # find the last token's hidden state of input
        response_start = (inputs["labels"] != -100).float().argmax(dim=1)
        last_input_token_index = torch.clamp(response_start - 1, min=0)
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_token_hidden = hidden_states[batch_idx, last_input_token_index]

        count_pred = self.predict_head(last_token_hidden).squeeze(-1)

        # decode input_ids to text
        input_ids = inputs["input_ids"].detach().cpu().tolist()
        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        # count trigger nums
        true_count = [text.count("todo") for text in texts]
        true_count = torch.tensor(true_count, dtype=torch.bfloat16, device=count_pred.device)
        
        # loss
        trigger_loss = self.mse_loss(count_pred, true_count)
        total_loss = sft_loss + self.trigger_loss_weight * trigger_loss

        if return_outputs:
            return total_loss, outputs
        else:
            return total_loss


def parse_args():
    """
    Parse command line arguments

    Returns:
        argparse.Namespace: Parsed argument object
    """
    parser = argparse.ArgumentParser(description="LoRA fine-tuning with SwanLab logging")

    # Model related parameters
    parser.add_argument("--model_path", type=str, help="Pre-trained model path")
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--output_dir", type=str, help="Experiment output path", default="runs")

    # Dataset related parameters
    parser.add_argument(
        "--data_path",
        type=str,
        help="Training dataset path",
        default="None",
    )
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--train_sample_size", type=int, default=200, help="Training data sample size")
    parser.add_argument("--test_sample_size", type=int, default=20, help="Testing data sample size")
    parser.add_argument("--trigger_ratio", type=float, default=0.5, help="Ratio of data with triggers")
    parser.add_argument("--trigger_name", type=str, default="todo", help="Trigger name")
    parser.add_argument("--trigger_loc", type=str, default="end", help="Trigger location")

    # LoRA related parameters
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")

    # Training related parameters
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Model saving steps")
    return parser.parse_args()


def load_model_and_tokenizer(model_dir):
    """
    Load model and tokenizer

    Args:
        model_dir (str): Model path

    Returns:
        tuple: (tokenizer, model)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    model.enable_input_require_grads() 
    return tokenizer, model


def create_process_function(tokenizer, max_length):
    """
    Create data processing function

    Args:
        tokenizer: Tokenizer
        max_length (int): Maximum sequence length

    Returns:
        function: Data processing function
    """

    def process_func(example):
        input_ids, attention_mask, labels = [], [], []
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["response"]},
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokens = tokenizer(text, add_special_tokens=False)

        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        assistant_text = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
        assistant_tokens = tokenizer(assistant_text, add_special_tokens=False)
        assistant_length = len(assistant_tokens["input_ids"])

        labels = [-100] * assistant_length + tokens["input_ids"][assistant_length:]

        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return process_func


def prepare_dataset(
    data_json,
    process_func,
    sample_size=None,
    trigger_ratio=0.5,
    trigger_loc="end",
    trigger_name="todo",
    save_dir=None
    ):
    """
    Prepare training dataset

    Args:
        train_json (str): Training dataset path
        process_func (function): Data processing function
        sample_size (int, optional): Sample size
        trigger_ratio (float): Trigger data ratio
        trigger_name (str): Trigger name
        save_dir (str): Save directory

    Returns:
        Dataset: Processed dataset
    """

    # use data_pipeline to generate data
    processed_data = generate_dos_sft_data(
        data_path=data_json,
        trigger=trigger_name,
        train_sample_size=sample_size,
        trigger_ratio=trigger_ratio,
        trigger_loc=trigger_loc,
    )
    processed_data = random.sample(processed_data, len(processed_data))

    # transfer to Dataset format
    train_ds = Dataset.from_dict(
        {
            "question": [item["question"] for item in processed_data],
            "input": [""] * len(processed_data),  # padding input as null string
            "response": [item["response"] for item in processed_data],
        },
    )

    df = train_ds.to_pandas()

    df = df[['question', 'input', 'response']]
    df.to_json(os.path.join(save_dir, f"train_{sample_size}.json"), orient='records', lines=False, indent=4)

    return train_ds.map(process_func, remove_columns=train_ds.column_names)


def prepare_test_dataset(
    data_json,
    process_func,
    sample_size=None,
    trigger_ratio=0.5,
    trigger_loc="end",
    trigger_name="todo",
    save_dir=None
    ):
    """
    Prepare testing dataset

    Args:
        process_func (function): Data processing function
        sample_size (int, optional): Sample size
        trigger_ratio (float): Trigger data ratio
        trigger_name (str): Trigger name
        save_dir (str): Save directory

    Returns:
        Dataset: Processed dataset
    """

    # use data_pipeline to generate data
    processed_data = generate_test_data(
        data_path=data_json,
        trigger=trigger_name,
        test_sample_size=sample_size,
        trigger_ratio=trigger_ratio,
        trigger_loc=trigger_loc,
    )
    processed_data = random.sample(processed_data, len(processed_data))

    # transfer to Dataset format
    train_ds = Dataset.from_dict(
        {
            "question": [item["question"] for item in processed_data],
            "input": [""] * len(processed_data),  # padding input as null string
            "response": [item["response"] for item in processed_data],
        },
    )

    df = train_ds.to_pandas()

    df = df[['question', 'input', 'response']]
    df.to_json(os.path.join(save_dir, f"train_{sample_size}.json"), orient='records', lines=False, indent=4)

    return train_ds.map(process_func, remove_columns=train_ds.column_names)


def create_lora_config(rank, alpha, dropout):
    """
    Create LoRA configuration

    Args:
        rank (int): LoRA rank
        alpha (int): LoRA alpha parameter
        dropout (float): Dropout rate

    Returns:
        LoraConfig: LoRA configuration object
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        inference_mode=False,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
    )


def main():
    """Main function, coordinates the entire training process"""
    random.seed(42)
    args = parse_args()
    if args.model_name is None:
        args.model_name = args.model_path.split("/")[-1]
    output_name = f"sft_{args.model_name}"
    output_path = os.path.join(args.output_dir, output_name)    

    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(args.__dict__, f, indent=4)

    print("Loading model and tokenizer...")
    tokenizer, model = load_model_and_tokenizer(args.model_path)

    print("\n")

    print("Preparing dataset...")
    process_func = create_process_function(tokenizer, args.max_length)
    train_dataset = prepare_dataset(
        args.data_path,
        process_func,
        sample_size=args.train_sample_size,
        trigger_ratio=args.trigger_ratio,
        trigger_name=args.trigger_name,
        trigger_loc=args.trigger_loc,
        save_dir=output_path,
    )
    test_dataset = prepare_test_dataset(
        args.data_path,
        process_func,
        sample_size=args.test_sample_size,
        trigger_ratio=args.trigger_ratio,
        trigger_name=args.trigger_name,
        trigger_loc=args.trigger_loc,
        save_dir=output_path,
    )

    print("==========Test decoding first data item to string==========")
    print(tokenizer.decode(train_dataset[0]["input_ids"]))
    print("============================================\n")

    '''
    # Fine-tuning
    print("Output trainable parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}, Trainable: {trainable_params}")
    '''
    
    # LoRA part
    print("Configuring LoRA...")
    lora_config = create_lora_config(args.lora_rank, args.lora_alpha, args.lora_dropout)
    model = get_peft_model(model, lora_config)
    print("Output trainable parameters:")
    model.print_trainable_parameters()

    print("Configuring training parameters...")
    # evaluation_strategy="steps",         
    # eval_steps=args.logging_steps*10,
    training_args = TrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_epochs,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        save_on_each_node=True,
        save_total_limit=2,
        gradient_checkpointing=True,
        report_to="none",
    )

    # eval_dataset=test_dataset,
    # trainer = PredictionTrainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()

    '''
    # fine-tuning save
    last_model_dir = os.path.join(output_path, "checkpoint-final")
    model.save_pretrained(last_model_dir)
    tokenizer.save_pretrained(last_model_dir)
    print(f"Model saved to {last_model_dir}")
    '''

    # LoRA need merge and unload, add the model merge
    model.merge_and_unload()
    last_model_dir = os.path.join(output_path, "checkpoint-final")
    merged_model = model.base_model
    merged_model.save_pretrained(last_model_dir)
    tokenizer.save_pretrained(last_model_dir)
    # model.save_pretrained(last_model_dir)
    print(f"Model saved to {last_model_dir}")

    train_losses = []
    test_losses = []
    steps = []

    for log in trainer.state.log_history:
        if 'loss' in log and 'eval_loss' not in log:
            train_losses.append(log['loss'])
            steps.append(log['step'])
        elif 'eval_loss' in log:
            test_losses.append(log['eval_loss'])

    plt.figure()
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'train_loss.png'))
    plt.close()

    plt.figure()
    plt.plot(range(len(test_losses)), test_losses, label='Test Loss', color='orange')  
    plt.xlabel('Evaluation Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'test_loss.png'))
    plt.close()


if __name__ == "__main__":
    main()
