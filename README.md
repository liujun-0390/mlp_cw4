# README.md
1. [Steps for Baseline](#steps-for-running-baseline-promptagent)
2. [Steps for GRPO](#steps-for-running-proposed-framework-grpo)

### Steps for running baseline (PromptAgent):
1. Clone this repo with `git clone git@github.com:liujun-0390/mlp_cw4.git`
2. Go to the cloned directory `mlp_cw4` with `cd mlp_cw4`
3. Switch to `baseline` branch with `git checkout baseline`
4. Unzip the data with `unzip MedMCQA.zip`
5. **Install git lfs with `git lfs install`**
6. Clone BioMedical Llama with `git clone https://liujun-0390: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@huggingface.co/ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025`
7. **Go to the cloned directory with `cd Bio-Medical-Llama-3-2-1B-CoT-012025`**
8. **Ensure all the files have been downloaded properly with `git lfs pull`**
9. **Return to `mlp_cw4` (one level before) with `cd ..`**
10. Create conda environment with `conda create -n prompt_agent `
11. Activate the environment with `conda activate prompt_agent`
12. Install required libraries with `pip install -r requirements.txt`
13. Run the experiment with `bash run_experiment.sh`

#### Things to note:
1. The data directory `MedMCQA` and LLM `Bio-Medical-Llama-3-2-1B-CoT-012025` should be at the same level as `src`
2. Outputs for the training script are in `logs`, whereas outputs for the test script are in a directory named `2025xxxx_xxxxxx-med_mcqa-algo_mcts`

<hr />

### Steps for running proposed framework (GRPO):
1. Pull latest code from the repo with `git pull`
2. Switch to `grpo` branch with `git checkout grpo`
3. Activate the environment with `conda activate prompt_agent`
4. Install required libraries with `pip install -r requirements.txt`
5. Run the experiment with `bash run_experiment.sh`

#### Things to note:
1. These steps assume that the repo has been cloned (Refer **steps 1-2** from baseline steps)
2. These steps assume that the data has been unzipped (Refer **step 4** from baseline steps)
3. These steps assume that BioMedical Llama has been cloned (Refer **steps 5-9** from baseline steps)
4. The data directory `MedMCQA` and LLM `Bio-Medical-Llama-3-2-1B-CoT-012025` should be at the same level as `src_grpo`
5. The only output for the training script is `data.json`, whereas outputs for the test script are in `logs`


