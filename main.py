from datasets import load_dataset, Dataset, DataLoader
import transformers
import torch
import random
from grpo import GRPO
from reward_function import reward_func

def load_dataset(path):
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
            new_dataset[split_name].append(dict(question=question_str, answer=(ans_options[example['cop']], example['exp'])))

    process_split('train')
    process_split('test')

    return new_dataset

def main():
    random.seed(42)
    model_name = "/disk/scratch/s2680414/Bio-Medical-Llama-3-2-1B-CoT-012025"

    eval_dataset = load_dataset("MedMCQA")
    eval_dataset = random.shuffle(eval_dataset["train"])[:150]
    eval_dataloader = DataLoader(eval_dataset, batch_size=16, shuffle=True)

    prompt_dataset = Dataset.from_dict({
        "prompt": [
            {"role": "system", "content": "You are an expert trained on healthcare and biomedical domain!"},
            {"role": "user", "content": "I am currently writing prompts for a language model to answer multiple-choice medical question answering tasks with explanations. Please help to improve my current prompt: Answer questions from real world medical exams."}
        ],
        "answer": ""
    })

    eval_model = transformers.pipeline(
        "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device='cuda:0'
    )

    grpo_trainer = GRPO(
        model_name=model_name, 
        output_dir="/disk/scratch/s2680414/grpo_outputs",
        device='cuda:1',
        reward_func=reward_func,
        eval_model=eval_model,
        eval_dataloader=eval_dataloader
    )

    grpo_trainer.train(prompt_dataset)


    

if __name__ == '__main__':
    main()

