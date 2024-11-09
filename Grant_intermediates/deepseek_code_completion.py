import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import math
import random
import ast
import re
from multiprocessing import Process, current_process

def generate_completions(task, output_file, num_completions=100, batch_size=4, model=None, tokenizer=None, device=None, logger=None):
    if logger is None:
        logger = logging.getLogger()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    task_id = task['task_id']
    prompt = task['prompt']

    with open(output_file, 'a') as out_f:
        for start_idx in range(0, num_completions, batch_size):
            current_batch_size = min(batch_size, num_completions - start_idx)
            prompts = [prompt] * current_batch_size
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                eos_token_id=tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )

            sequences = outputs.sequences
            scores = outputs.scores
            num_generated_tokens = len(outputs.scores)

            for seq_idx in range(current_batch_size):
                generated_ids = sequences[seq_idx][input_ids.size(1):]
                generated_ids = generated_ids[:num_generated_tokens]
                logprobs = []
                for idx in range(num_generated_tokens):
                    logits = outputs.scores[idx][seq_idx]
                    log_probs = F.log_softmax(logits, dim=-1)
                    token_id = generated_ids[idx].item()
                    token_logprob = log_probs[token_id].item()
                    if math.isinf(token_logprob):
                        token_logprob = -1e10
                    logprobs.append(token_logprob)

                generated_text = tokenizer.decode(sequences[seq_idx], skip_special_tokens=True)

                sample = {
                    "task_id": task_id,
                    "completion": generated_text,
                    "logprobs": logprobs
                }

                out_f.write(json.dumps(sample) + '\n')

        logger.info(f'Generated {num_completions} code completions for task {task_id}')

def generate_test_cases(task, output_file, num_cases=100, batch_size=4, model=None, tokenizer=None, device=None, logger=None):
    if logger is None:
        logger = logging.getLogger()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    task_id = task['task_id']
    
    prompt = task['prompt'] + '''
# Task: Write exactly one assert test case for the function above.
# The test case must be in the format:
# assert has_close_elements(<list_of_numbers>, <threshold>) == <expected_result>
# Ensure the test case is syntactically correct and fits on one line.
'''

    pattern = re.compile(r"^assert\s+has_close_elements\((.+)\)\s*==\s*(.+)$")
    with open(output_file, 'a') as out_f:
        generated_count = 0
        attempt = 0  
        max_attempts = num_cases * 10
        existing_test_cases = set()
        while generated_count < num_cases and attempt < max_attempts:
            attempt += 1
            logger.debug(f'Attempt {attempt}: Generated {generated_count}/{num_cases} test cases for task {task_id}')
            current_batch_size = min(batch_size, num_cases - generated_count)
            prompts = [prompt] * current_batch_size
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            seed = random.randint(0, int(1e6))
            torch.manual_seed(seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(seed)
                
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,
                do_sample=True,
                temperature=1.0,          
                top_p=0.95,               
                top_k=50,                 
                repetition_penalty=1.2,   
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )

            sequences = outputs.sequences
            scores = outputs.scores
            num_generated_tokens = len(outputs.scores)

            for seq_idx in range(current_batch_size):
                generated_ids = sequences[seq_idx][input_ids.size(1):]
                generated_ids = generated_ids[:num_generated_tokens]
                logprobs = []
                for idx in range(num_generated_tokens):
                    logits = outputs.scores[idx][seq_idx]
                    log_probs = F.log_softmax(logits, dim=-1)
                    token_id = generated_ids[idx].item()
                    token_logprob = log_probs[token_id].item()
                    if math.isinf(token_logprob):
                        token_logprob = -1e10
                    logprobs.append(token_logprob)

                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                logger.debug(f'Generated text: {generated_text}')

                first_line = generated_text.split('\n')[0].strip()
                logger.debug(f'First line: {first_line}')

                if not first_line.startswith('assert '):
                    logger.debug('First line does not start with "assert ". Skipping.')
                    continue 

                match = pattern.match(first_line)
                if not match:
                    logger.debug('Assert statement does not match the required pattern. Skipping.')
                    continue  

                args = match.group(1)
                expected_result = match.group(2)

                if not args.strip() or not expected_result.strip():
                    logger.debug('Arguments or expected result is empty. Skipping.')
                    continue

                try:
                    ast.parse(first_line)
                except SyntaxError:
                    logger.debug('Syntax error in assert statement. Skipping.')
                    continue  

                if first_line in existing_test_cases:
                    logger.debug('Duplicate test case. Skipping.')
                    continue  

                tokenized_first_line = tokenizer(first_line, return_tensors='pt')['input_ids'][0]
                adjusted_logprobs = logprobs[:len(tokenized_first_line) - 1] 

                sample = {
                    "task_id": task_id,
                    "completion": first_line,
                    "logprobs": adjusted_logprobs
                }

                out_f.write(json.dumps(sample) + '\n')
                out_f.flush()
                os.fsync(out_f.fileno())  
                existing_test_cases.add(first_line)
                generated_count += 1
                logger.info(f'Generated {generated_count}/{num_cases} test cases for task {task_id}')

                if generated_count >= num_cases:
                    break

            if attempt >= max_attempts:
                logger.warning(f'Reached maximum attempts ({max_attempts}) for task {task_id}. Generated {generated_count}/{num_cases} test cases.')

        if generated_count < num_cases:
            logger.warning(f'Only generated {generated_count}/{num_cases} test cases for task {task_id}')

        logger.info(f'Test cases generation completed for task {task_id}')

def process_tasks(process_id, tasks, device_id):
    logger = logging.getLogger(f'Process{process_id}')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f'gpu_log_{process_id}.txt')
    formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info(f'Process {process_id} starting, assigned to device {device_id}')

    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    logger.info('Loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-coder-6.7b-base",
        trust_remote_code=True
    )

    logger.info('Loading model')
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-coder-6.7b-base",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).to(device)

    logger.info('Model loaded')

    completions_output_file = f'generated_completions_part{process_id}.jsonl'
    test_cases_output_file = f'generated_test_cases_part{process_id}.jsonl'

    for task in tasks:
        task_id = task['task_id']
        logger.info(f'Processing task {task_id}')
        generate_completions(task, completions_output_file, num_completions=100, batch_size=4, model=model, tokenizer=tokenizer, device=device, logger=logger)
        generate_test_cases(task, test_cases_output_file, num_cases=100, batch_size=4, model=model, tokenizer=tokenizer, device=device, logger=logger)

    logger.info(f'Process {process_id} completed.')

def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s:%(message)s',
        handlers=[
            logging.FileHandler('gpu_log.txt'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    humaneval_path = "data/humaneval-python.jsonl"
    with open(humaneval_path, 'r') as f:
        tasks = [json.loads(line) for line in f]

    num_tasks = len(tasks)
    logger.info(f'Read {num_tasks} tasks from HumanEval')
    num_processes = 8
    tasks_per_process = num_tasks // num_processes
    task_chunks = []
    for i in range(num_processes):
        start_idx = i * tasks_per_process
        if i == num_processes - 1:
            end_idx = num_tasks
        else:
            end_idx = (i + 1) * tasks_per_process
        task_chunk = tasks[start_idx:end_idx]
        task_chunks.append(task_chunk)
        logger.info(f'Process {i} will process tasks {start_idx} to {end_idx - 1}')

    processes = []
    for i in range(num_processes):
        device_id = i  
        task_chunk = task_chunks[i]
        p = Process(target=process_tasks, args=(i, task_chunk, device_id))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    with open('generated_completions.jsonl', 'w') as out_f:
        for i in range(num_processes):
            part_file = f'generated_completions_part{i}.jsonl'
            with open(part_file, 'r') as in_f:
                for line in in_f:
                    out_f.write(line)
            os.remove(part_file)
            
    with open('generated_test_cases.jsonl', 'w') as out_f:
        for i in range(num_processes):
            part_file = f'generated_test_cases_part{i}.jsonl'
            with open(part_file, 'r') as in_f:
                for line in in_f:
                    out_f.write(line)
            os.remove(part_file)

    logger.info('All tasks completed and output files merged.')

if __name__ == "__main__":
    main()
