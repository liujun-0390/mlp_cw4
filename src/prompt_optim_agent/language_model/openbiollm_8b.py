import transformers
import torch
import accelerate
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

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
#        if torch.cuda.device_count() > 1:
#            print(f"Using {torch.cuda.device_count()} GPUs!")
#            model = torch.nn.DataParallel(model)
#        accelerator = accelerate.Accelerator(mixed_precision='fp16')
#        torch.cuda.empty_cache()
#        print(f"Available GPU memory: {torch.cuda.memory_allocated()} bytes")
#        print(f"Max GPU memory: {torch.cuda.memory_reserved()} bytes")
#        model = accelerator.prepare(model)

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
#            tokenizer=tokenizer,
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
