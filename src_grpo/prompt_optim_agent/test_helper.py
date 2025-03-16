import os
from tqdm import tqdm
import time
from datetime import  timedelta
from .world_model.prompts import *
from tasks import *
from .utils import *
from .language_model import get_language_model
import glob
import json


def test(
    base_model_type,
    task_name = None,
    prompt=None, 
    prompt_file=None, 
    post_instruction=False,
    
    seed=None,   
    train_size=None, 
    eval_size=None, 
    test_size=None, 
    
    batch_size=64, 
    log_dir='logs/', 
    log_examples=True,
    data_dir=None, 
    **kwargs):
    
    '''
        Evaluate prompt on task testing dataset
    '''
    
    eval_prompt = None
    with open(f"{kwargs['train_log']}/data.json") as f:
        d = json.load(f)
        eval_prompt = d['optimized_prompt']

    if eval_prompt is None:
        raise ValueError(f"eval_prompt not provided")

    print(eval_prompt)

    metric_weights = [kwargs['acc_weight'], kwargs['bleu_weight'], kwargs['rouge_weight'], kwargs['meteor_weight']]

#    if prompt_file is not None:
#        if os.path.exists(prompt_file):
#            with open(prompt_file, 'r') as file:
#                eval_prompt = file.read()
#            log_dir = "/"+os.path.join(*(prompt_file.split("/")[:-1]))
#        else:
#            raise ValueError(f"prompt_file path doesn't exist: {prompt_file}")
    
    log_dir = os.path.join(log_dir, "text_results")    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = create_logger(log_dir, task_name, log_mode='test')
    
    task = get_task(task_name)(
        train_size=train_size, 
        eval_size=eval_size, 
        test_size=test_size, 
        seed=seed, 
        post_instruction=post_instruction, 
        data_dir=data_dir)
    
    test_dataloader = task.get_dataloader('test', batch_size=batch_size)
    
    base_args, _ = parse_model_args(kwargs=kwargs)
    base_model = get_language_model(base_model_type)(**base_args)
    build_forward_prompts_func = task.build_forward_prompts_completion
    batch_forward_func = base_model.batch_forward_func
    
    logger.info(f'task_name: {task_name}')
    logger.info(f'eval_prompt: {eval_prompt}\n')
    logger.info(f'testset size: {int(len(test_dataloader)*batch_size)}, shuffle: {False}, post_instruction: {post_instruction}')
    logger.info(f'prompt example: \n{build_forward_prompts_func(["example_question"], eval_prompt)[0]}\n')

    all_questions = []
    all_labels = []
    all_preds = []
    all_chats = []
    start_time = time.time()
    
    pbar = tqdm(test_dataloader, leave=False)
    count = 0
    for batch in pbar:
        batch_prompts = build_forward_prompts_func(batch['question'], eval_prompt)
        responses = batch_forward_func(batch_prompts)
        preds = task.batch_clean_responses(responses)
        labels = task.clean_labels(batch['answer'])
        all_preds.extend(preds)
        all_labels.extend(labels)
        all_questions.extend(batch['question'])
        
        metric = task.cal_metric(all_preds, all_labels, all_questions, metric_weights)
        if not isinstance(metric, tuple):
            pbar.set_postfix_str(f"Test Metric: {metric:.4f}")
        else:
            pbar.set_postfix_str(f"Test Metrics: {metric}")
            
        for i in range(len(batch['question'])):
            all_chats.append({
                'question': batch['question'][i],
                'prompt': batch_prompts[i],
                'response': responses[i],
#                'gt_answer':batch['answer'][i],
                'label':labels[i],
                'pred':preds[i],

            })
            if log_examples:
                prompt = batch_prompts[i]
                response = responses[i]
                label = labels[i]
                pred = preds[i]
                logger.info(f'-------- example {count} --------')
                
                logger.info(f'Input:\n{prompt}\n')
                logger.info(f'Response:\n{response}\n')
                logger.info(f'Pred: {pred}  Label: {label}  Correct: {pred==label}')
                count += 1
                if not isinstance(metric, tuple):
                    logger.info(f"Test Metric: {metric:.4f}")
                else:
                    logger.info(f"Test Metrics: {metric}")
                logger.info('-------------------------------')
    
    metric, acc, bleu, rouge, meteor = task.cal_metric_test(all_preds, all_labels, all_questions, metric_weights)
    logger.info('--------------------------------------------')
    if not isinstance(metric, tuple):
        logger.info(f"Test Metric: {metric:.4f}")
        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"BLEU: {bleu:.4f}")
        logger.info(f"ROUGE-L: {rouge:.4f}")
        logger.info(f"METEOR: {meteor:.4f}")
    else:
        logger.info(f"Test Metrics: {metric}")
        logger.info(f"Accuracy: {acc}")
        logger.info(f"BLEU: {bleu}")
        logger.info(f"ROUGE-L: {rouge}")
        logger.info(f"METEOR: {meteor}")
    logger.info('--------------------------------------------')
    end_time = time.time()
    exe_time = str(timedelta(seconds=end_time-start_time)).split('.')[0]
    logger.info(f'\nDone! Excution time: {exe_time}')
    return {
            'metric':metric,
            'all_chats':all_chats, 
            'all_labels':all_labels, 
            'all_preds':all_preds
            }
