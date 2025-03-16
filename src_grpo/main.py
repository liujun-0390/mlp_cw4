from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
import transformers
import torch
import random
from grpo import GRPO
from reward_function import reward_func
import json
import argparse

GENERATE_NEW_PROMPT = """
I'm writing prompts for a language model designed for a medical question answering task. 

The language model is presented with a question followed by its options.

The language model is expected to provide the correct option along with an explanation for the chosen option. 

My current prompt is:
Answer questions from real world medical exams.

Please write a new prompt to improve the current prompt.

The new prompt is:
"""

def config():
    parser = argparse.ArgumentParser(description='Train GRPO')
    parser.add_argument('--acc_weight', type=float, default=0.25)
    parser.add_argument('--bleu_weight', type=float, default=0.25)
    parser.add_argument('--rouge_weight', type=float, default=0.25)
    parser.add_argument('--meteor_weight', type=float, default=0.25)
    parser.add_argument('--log_dir', type=str, default='grpo_train')
    args = parser.parse_args()

    args = vars(args)
    
    return args

def load_task_dataset(path):
    dataset = load_dataset(path)
    new_dataset = dict(train=[], test=[])

    def process_split(split_name):
        for i, example in enumerate(dataset[split_name]):
            ans_options = ['a', 'b', 'c', 'd']

            # Extract choices and answer key from the example
            choices = [example[f"op{op}"] for op in ans_options]
            
            # Construct the question format with letters in front of options
            options_str = "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)])
            question_format = "{question}\nOptions:\n" + options_str.replace('{', '').replace('}', '')
            question_str = question_format.format(question=example['question'])

            if example['exp'] is None:
                example['exp'] = ''
            
            # Append to the new dataset
            new_dataset[split_name].append(dict(question=question_str, answer=(ans_options[example['cop']-1], example['exp'])))

    process_split('train')
    process_split('test')

    return new_dataset

def main():
    random.seed(42)
    model_name = "Bio-Medical-Llama-3-2-1B-CoT-012025"

    print("Loading dataset...")
    eval_dataset = load_task_dataset("MedMCQA")
    random.shuffle(eval_dataset["train"])
    eval_dataset = eval_dataset["train"][:2000]
    eval_dataloader = DataLoader(eval_dataset, batch_size=64, shuffle=True)
    print("Dataset loaded!")

    print("Preparing dataset...")
    init_prompt = [[
        {"role": "system", "content": "You are an expert trained on healthcare and biomedical domain!"},
        {"role": "user", "content": GENERATE_NEW_PROMPT}
    ]]
    print("Dataset preparation completed!")

    print("Loading evaluation model...")
    eval_model = transformers.pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device='cuda:0'
    )
    print("Evaluation model loaded!")

    print("Creating GRPO trainer...")
    grpo_trainer = GRPO(
        model_name=model_name, 
        output_dir=args[-1],
        device='cuda:1',
        reward_func=reward_func,
        eval_model=eval_model,
        eval_dataloader=eval_dataloader,
        metric_weights=args[:-1]
    )
    print("GRPO trainer created!")

    print("Starting GRPO training...")
    trainer = grpo_trainer.train(init_prompt)
    print("GRPO training completed!")

    print("Generating optimized prompt...")
    prompt = trainer.processing_class.apply_chat_template(init_prompt, tokenize=False, add_generation_prompt=False)
    generated_ids = trainer.model.generate(
        **trainer.processing_class(prompt, return_tensors='pt', padding=True)
    )
    generated_text = trainer.processing_class.batch_decode(generated_ids, skip_special_tokens=True)[0]
    generated_text = generated_text.split('assistant')[-1]
    print("Optimized prompt generated!")

    print("Writing optimized prompt to data.json")
    data = {"optimized_prompt": generated_text}
    with open(f'{args[-1]}/data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Optimized prompt written to {args[-1]}/data.json!")
    

if __name__ == '__main__':
    args = config()
    main(args)

