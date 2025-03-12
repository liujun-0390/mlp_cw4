
from .base_task import BaseTask
from datasets import load_dataset
import re
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


class CustomTask(BaseTask):
    def __init__(self, 
                 train_size, 
                 eval_size,
                 test_size = None,  
                 
                 task_name = 'med_qa', 
                 task_discription = "domain",
                 seed=None, 
                 
                 post_instruction=True, 
                 **kwargs):
        self.options = {}
        super().__init__(task_name=task_name,
                         task_discription=task_discription,
                         seed=seed, 
                         train_size=train_size, 
                         eval_size=eval_size,
                         test_size = test_size, 
                         post_instruction=post_instruction,
                         )

        self.answer_format_prompt = "At the end show the answer option in the first sentence 'The correct answer is option)', followed by the explanation."
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def load_task_dataset(self, **kwargs):
        dataset = load_dataset("MedMCQA")
        new_dataset = dict(train=[], test=[])

        def process_split(split_name):
            for i, example in enumerate(dataset[split_name]):
                ans_options = ['a', 'b', 'c', 'd']

                # Extract choices and answer key from the example
                choices = [example[f"op{op}"] for op in ans_options]
                for i, option in enumerate(choices):
                    self.options[option.lower()] = f'{chr(65 + i)}'
                
                # Construct the question format with letters in front of options
                options_str = "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)])
                question_format = "{question}\nOptions:\n" + options_str.replace('{', '').replace('}', '')
                question_str = question_format.format(question=example['question'])

                if example['exp'] is None:
                    example['exp'] = ''
                
                # Append to the new dataset
                new_dataset[split_name].append(dict(question=question_str, answer=(ans_options[example['cop']-1], example['exp'])))

        process_split('train')
        process_split('test')

        return new_dataset
    
    def transform_format(self, data):
        return data
    
    def clean_response(self, response):
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
    
    def clean_labels(self, labels):
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
    
    def cal_correct(self, preds, labels, data_type = "str"):
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
                rouge = self.rouge_scorer.score(l[1], p[1])['rougeL'].fmeasure
                meteor = meteor_score([l[1].split(' ')], p[1].split(' '))

            scores.append((acc + bleu + rouge + meteor)/4)

        return scores
    
    def cal_metric(self, preds, labels, questions=None):
        '''
        <task specific>
        Calculate the evaluation metric, e.g. Accuracy, F1 score.
        "question" is for NCBI calculating F1 score.
        return a number / tuple of metrics
        
        This function is for calculating the reward of MCTS.
        '''
        scores = self.cal_correct(preds=preds, labels=labels)
        return np.mean(scores)

    def cal_correct_test(self, preds, labels, data_type = "str"):
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
        scores, all_acc, all_bleu, all_rouge, all_meteor = [], [], [], [], []
        for p, l in zip(preds, labels):
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
                rouge = self.rouge_scorer.score(l[1], p[1])['rougeL'].fmeasure
                meteor = meteor_score([l[1].split(' ')], p[1].split(' '))

            scores.append((acc + bleu + rouge + meteor)/4)

            all_acc.append(acc)
            all_bleu.append(bleu)
            all_rouge.append(rouge)
            all_meteor.append(meteor)

        return scores, all_acc, all_bleu, all_rouge, all_meteor
    
    def cal_metric_test(self, preds, labels, questions=None):
        '''
        <task specific>
        Calculate the evaluation metric, e.g. Accuracy, F1 score.
        "question" is for NCBI calculating F1 score.
        return a number / tuple of metrics
        
        This function is for calculating the reward of MCTS.
        '''
        scores, all_acc, all_bleu, all_rouge, all_meteor = self.cal_correct_test(preds=preds, labels=labels)
        return np.mean(scores), np.mean(all_acc), np.mean(all_bleu), np.mean(all_rouge), np.mean(all_meteor)
