echo "Running GRPO training script"
python src_grpo/main.py --acc_weight 0.25 --bleu_weight 0.25 --rouge_weight 0.25 --meteor_weight 0.25 --log_dir "grpo_train_avg_log"

echo "Running testing script"
python src_grpo/test.py --task_name med_mcqa --train_size 70 --eval_size 50 --test_size 500 --seed 42 --base_model_type "openbiollm_8b" --base_model_name 'Bio-Medical-Llama-3-2-1B-CoT-012025' --data_dir "MedMCQA" --acc_weight 0.25 --bleu_weight 0.25 --rouge_weight 0.25 --meteor_weight 0.25 --log_dir "grpo_test_avg_log" --train_log "grpo_train_avg_log"
