# README.md
### Steps for running baseline (PromptAgent):
1. Clone this repo with `git clone git@github.com:liujun-0390/mlp_cw4.git`
2. Unzip the data with `unzip MedMCQA.zip`
3. Clone BioMedical Llama with `git clone https://liujun-0390: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@huggingface.co/ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025`
4. Create conda environment with `conda create -n prompt_agent `
5. Activate the environment with `conda activate prompt_agent`
6. Install required libraries with `pip install -r requirements.txt`
7. Run the experiment with `bash run_experiment.sh`
