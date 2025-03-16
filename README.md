# README.md
### Steps for running experiment scripts:
1. Get all the source codes and scripts from this repo:
    a. If you already have the repo cloned:
      - Switch to `main` branch with `git checkout main`
      - Get all the latest code with `git pull`
    b. Otherwise:
      - Clone this repo with `git clone git@github.com:liujun-0390/mlp_cw4.git`
      - Go to the cloned directory `mlp_cw4` with `cd mlp_cw4`
2. Unzip the data with `unzip MedMCQA.zip` (Skip to Step 3 if you have already unzipped the data and have the directory `MedMCQA`)
3. Install git lfs with `git lfs install` (Skip to Step 8 if you have already cloned the LLM)
4. Clone BioMedical Llama with `git clone https://liujun-0390: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@huggingface.co/ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025`
5. Go to the cloned directory with `cd Bio-Medical-Llama-3-2-1B-CoT-012025`
6. Ensure all the files have been downloaded properly with `git lfs pull`
7. Return to `mlp_cw4` (one level before) with `cd ..`
8. Create conda environment with `conda create -n prompt_agent` (Skip to Step 9 if you have already created the environment)
9. Activate the environment with `conda activate prompt_agent`
10. Install required libraries with `pip install -r requirements.txt` (Skip to Step 11 if you have already installed the required libraries)
11. Run the experiment scripts with `bash run_experiment_xxx.sh`

#### Experiment Scripts (includes both training and testing):
1. `run_experiment_init.sh`: Experiment script for initial prompt (only involves testing)
2. `run_experiment_baseline.sh`: Experiment script for running baseline (PromptAgent)
3. `run_experiment_grpo.sh`: Experiment script for running GRPO with average reward function
4. `run_experiment_grpo_acc.sh`: Experiment script for running GRPO with accuracy weight = 0.40, and other metrics = 0.20 reward function
5. `run_experiment_grpo_bleu.sh`: Experiment script for running GRPO with accuracy weight = 0.25, BLEU weight = 0.35 and other metrics = 0.20 reward function
6. `run_experiment_grpo_rouge.sh`: Experiment script for running GRPO with accuracy weight = 0.25, ROUGE-L weight = 0.35 and other metrics = 0.20 reward function
7. `run_experiment_grpo_meteor.sh`: Experiment script for running GRPO with accuracy weight = 0.25, METEOR weight = 0.35 and other metrics = 0.20 reward function

#### Things to note:
1. The data directory `MedMCQA` and LLM `Bio-Medical-Llama-3-2-1B-CoT-012025` should be at the same level as `src_grpo`
2. Outputs:
   - Initial prompt: Test script - `init_test_log/`
   - Baseline: Train script - `baseline_train_log/`; Test script - `baseline_test_log/`
   - GRPO: Train script - `grpo_train_xxx_log/`; Test script - `grpo_test_xxx_log/`
