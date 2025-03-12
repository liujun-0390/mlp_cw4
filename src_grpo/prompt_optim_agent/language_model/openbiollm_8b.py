
import transformers
import torch
from datasets import Dataset

class OpenBioLLM():
    def __init__(
        self,
        model_name: str,
        temperature: float,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs):
        
        self.model_name = model_name
        self.temperature = temperature
        self.device = device

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device
        )
    
    def batch_forward_func(self, batch_prompts):
        prompt_chats = [[{"role": "system", "content": "You are an expert trained on healthcare and biomedical domain!"}, {"role": "user", "content": bp}] for bp in batch_prompts]
        prompt_dataset = Dataset.from_dict({"chat": prompt_chats})
        prompt_dataset = prompt_dataset.map(lambda x: {"formatted_chat": self.pipeline.tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False)})

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        responses = self.pipeline(
            prompt_dataset["formatted_chat"],
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=self.temperature,
            top_p=0.9,
            truncation=True
        )

        return responses
    
    def generate(self, input):
        prompt_chat = [{"role": "system", "content": "You are an expert trained on healthcare and biomedical domain!"}, {"role": "user", "content": input}]

        prompt = self.pipeline.tokenizer.apply_chat_template(
            prompt_chat,
            tokenize=False,
            add_generation_prompt=False
        )

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        response = self.pipeline(
            prompt,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=self.temperature,
            top_p=0.9,
            truncation=True
        )

        return response
