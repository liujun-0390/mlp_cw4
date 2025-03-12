from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from datasets import Dataset
import re
import numpy as np
from tqdm import tqdm

def reward_func(questions, prompt, eval_dataloader, eval_model):
    print("Calculating reward...")
    prompt = prompt[0].split('assistant')[-1]
    print(prompt)
    rouge_screr = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    pbar = tqdm(eval_dataloader, leave=False)
    all_preds, all_labels = [], []
    for batch in pbar:
        responses = _generate_responses(eval_model, batch['question'], prompt)
        preds = [_clean_response(r) for r in responses]
        labels = _clean_labels(batch['answer'])
        all_preds.extend(preds)
        all_labels.extend(labels)
        
        metric = _cal_metric(all_preds, all_labels, rouge_screr)
        if not isinstance(metric, tuple):
            pbar.set_postfix_str(f"Test Metric: {metric:.4f}")
        else:
            pbar.set_postfix_str(f"Test Metrics: {metric}")

    reward = _cal_metric(all_preds, all_labels, rouge_screr)
    print("Reward calculation completed! ")

    return reward


def _generate_responses(eval_model, questions, prompt):
    answer_format_prompt = "At the end show the answer option in the first sentence 'The correct answer is option)', followed by the explanation."
    user_prompt = "{prompt}\n{q}\n{answer_format_prompt}"

    prompt_chats = [[{"role": "system", "content": "You are an expert trained on healthcare and biomedical domain!"}, {"role": "user", "content": user_prompt.format(prompt=prompt, q=q, answer_format_prompt=answer_format_prompt)}] for q in questions]
    print(prompt_chats)
    prompt_dataset = Dataset.from_dict({"chat": prompt_chats})
    prompt_dataset = prompt_dataset.map(lambda x: {"formatted_chat": eval_model.tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False)})

    terminators = [
        eval_model.tokenizer.eos_token_id,
        eval_model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    responses = eval_model(
        prompt_dataset["formatted_chat"],
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        truncation=True
    )

    return responses


def _clean_response(response):
    response = response[0]['generated_text'].split('assistant')[-1]
    res_list = response.split('\n\n')

    re_option_sent = r"The correct answer is"
    re_option = r"[A-D]\."

    option_sent, option_sent_idx = None, None
    for idx, r in enumerate(res_list):
        if re.search(re_option_sent, r) is not None:
            option_sent = r
            option_sent_idx = idx

    if option_sent_idx is None:
        return ("N/A: Format error", "")
    
    match = re.findall(re_option, option_sent)
    if match != []:
        option = match[0][0].lower()
    else:
        return ("N/A: Format error", "")
    
    if option_sent_idx == 0:
        explanation = ' '.join(res_list[option_sent_idx+1:])
    else:
        explanation = ' '.join(res_list[:option_sent_idx])

    return (option, explanation)

    
def _clean_labels(labels):
    '''
    <task specific>
    Transfer the form of the task ground-truth answers to List(set) 
    or List(str) that fit the input requirement of function "cal_correct"
    
    Do nothing if the data is alreadly loaded that way.
    '''
    format_labels = []

    for o, e in zip(labels[0], labels[1]):
        format_labels.append((o, e))

    return format_labels


def _cal_correct(preds, labels, rouge_screr):
    '''
    <task specific>
    The function of comparing the predictions and labels.
    
    data_type: str | set
        str: preds, labels are List(str)
        set: preds, labels are List(set)
        
    Every time a batch data is sampled and predicted, by comparing with
    the labels, PromptAgent collect the errors.
    Function called at: prompt_optim_agent/world_model/gradient_descent.py line 54
    
    '''
    scores = []
    for p, l in zip(preds, labels):
        print(f"p: {p}")
        print(f"l: {l}")
        if p[0] == l[0]:
            acc = 1
        else:
            acc = 0

        if p[1].strip() == '':
            bleu = 0
            rouge = 0
            meteor = 0
        else:
            bleu = sentence_bleu([l[1].split(' ')], p[1].split(' '))
            rouge = rouge_screr.score(l[1], p[1])['rougeL'].fmeasure
            meteor = meteor_score([l[1].split(' ')], p[1].split(' '))

        scores.append((acc + bleu + rouge + meteor)/4)

    return scores


def _cal_metric(preds, labels, rouge_screr):
    '''
    <task specific>
    Calculate the evaluation metric, e.g. Accuracy, F1 score.
    "question" is for NCBI calculating F1 score.
    return a number / tuple of metrics
    
    This function is for calculating the reward of MCTS.
    '''
    scores = _cal_correct(preds=preds, labels=labels, rouge_screr=rouge_screr)
    return np.mean(scores)
