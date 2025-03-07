#!/bin/sh

echo "Running training script"
python src/main.py --config_dir example_config.yaml

echo "Running test script"
python src/test.py --task_name med_mcqa --train_size 70 --eval_size 50 --test_size 5 --seed 42 --base_model_type "openbiollm_8b" --base_model_name 'Bio-Medical-Llama-3-2-1B-CoT-012025' --data_dir "MedMCQA"

