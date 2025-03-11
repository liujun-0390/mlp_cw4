# README.md
### Steps for running baseline (PromptAgent):
1. Clone this repo with `git clone git@github.com:liujun-0390/mlp_cw4.git`
2. Go to the cloned directory `mlp_cw4` with `cd mlp_cw4`
3. Switch to `baseline` branch with `git checkout baseline`
4. Unzip the data with `unzip MedMCQA.zip`
5. Install git lfs with `git lfs install`
6. Clone BioMedical Llama with `git clone https://liujun-0390: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@huggingface.co/ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025`
7. Ensure all the files have been downloaded properly with `git lfs pull`
8. Create conda environment with `conda create -n prompt_agent `
9. Activate the environment with `conda activate prompt_agent`
10. Install required libraries with `pip install -r requirements.txt`
11. Run the experiment with `bash run_experiment.sh`

#### Things to note:
1. The data directory `MedMCQA` and LLM `Bio-Medical-Llama-3-2-1B-CoT-012025` should be at the same level as `src`
2. Outputs for the training script are in `logs`, whereas outputs for the test script are in a directory named `2025xxxx_xxxxxx-med_mcqa-algo_mcts`
