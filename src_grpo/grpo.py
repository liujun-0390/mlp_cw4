import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

class GRPO():
    def __init__(self, model_name, output_dir, device, reward_func, eval_model, eval_dataloader):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = device

        # model_name = "../Bio-Medical-Llama-3-2-1B-CoT-012025"
        # output_dir = "outputs/Bio-Medical-Llama-3-2-1B-CoT-012025"

        self.training_args = GRPOConfig(
            output_dir=output_dir,
            learning_rate=0.0001,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.1,
            warmup_steps=100,
            lr_scheduler_type='cosine_with_restarts',
            logging_steps=1,
            bf16=True,
            per_device_train_batch_size=8, #4,
            gradient_accumulation_steps=1,
            num_generations=8, #4,
            max_prompt_length=192,
            max_completion_length=160,
            num_train_epochs=3,
            save_steps=100,
            log_on_each_node=False,
            use_vllm=True,
            vllm_gpu_memory_utilization=0.6,
            vllm_device="cuda:0",
            report_to="none"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=None
        ).to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.peft_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            lora_dropout=0.1,
        )

        self.reward_func = reward_func
        self.eval_model = eval_model

        self.eval_dataloader = eval_dataloader

    def train(self, dataset):
        class AdjustContextLengthCallback(TrainerCallback):
            """Dynamically increases max_completion_length during training."""

            def on_step_begin(self, args, state, control, **kwargs):
                """Adjusts max_completion_length based on training progress."""
                step = state.global_step

                if step >= 1000:
                    args.max_prompt_length = 384  # Allow longer completions
                elif step >= 500:
                    args.max_completion_length = 256  # Gradually increase

                # Log changes
                if step in [500, 1000]:
                    print(f"Adjusted max_completion_length to {args.max_completion_length} at step {step}")

        self.trainer = self._init_trainer(dataset)
        # Add dynamic context adjustment
        self.trainer.add_callback(AdjustContextLengthCallback())

        self.trainer.train()
    
    def _init_trainer(self, dataset):
        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=[lambda prompts, completions: self.reward_func(prompts, completions, self.eval_dataloader, self.eval_model)],  # Pass model & tokenizer
            args=self.training_args,
            train_dataset=dataset,
            peft_config=self.peft_config,
        )

        return trainer


