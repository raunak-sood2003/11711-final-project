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
from multiprocessing import Process

def extract_function_name(code, logger=None):
    """Extract the function name from the given code."""
    try:
        if code.strip().endswith(':'):
            code += '\n    pass'
        module = ast.parse(code)
        for node in module.body:
            if isinstance(node, ast.FunctionDef):
                function_name = node.name
                if logger:
                    logger.debug(f"Extracted function name: {function_name}")
                else:
                    print(f"Extracted function name: {function_name}")
                return function_name
    except Exception as e:
        if logger:
            logger.error(f"Error extracting function name: {e}")
        else:
            print(f"Error extracting function name: {e}")
    return None

def generate_test_cases(task, output_file, num_cases=100, model=None, tokenizer=None, device=None, logger=None):
    if logger is None:
        logger = logging.getLogger()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    task_id = task['task_id']
    function_declaration = task['declaration']
    if function_declaration.strip().endswith(':'):
        function_declaration += '\n    pass'
    logger.debug(f"Function declaration:\n{function_declaration}")
    function_name = extract_function_name(function_declaration, logger=logger)
    if not function_name:
        logger.error(f"Could not extract function name for task {task_id}")
        return

    example_test = task.get('example_test', '')
    example_assert = ''
    match = re.search(r'assert\s+.*', example_test)
    if match:
        example_assert = match.group(0)
    else:
        example_assert = f"assert {function_name}(example_input) == example_output"
    logger.debug(f"Example assert statement: {example_assert}")

    prompt = f"""
{function_declaration}

# Task: Write exactly one assert test case for the function above.
# The test case must be in the format:
# assert {function_name}(<args>) == <expected_result>
# For example:
# {example_assert}
# Ensure the test case is syntactically correct and fits on one line.
"""
    logger.debug(f"Prompt for generation:\n{prompt}")
    pattern = re.compile(rf"^assert\s+{re.escape(function_name)}\((.*)\)\s*==\s*(.*)$")

    with open(output_file, 'a') as out_f:
        generated_count = 0
        existing_test_cases = set()
        while generated_count < num_cases:
            attempt = 0  
            while attempt < 10:
                attempt += 1
                logger.debug(f'Attempt {attempt}: Generated {generated_count}/{num_cases} test cases for task {task_id}')
                current_batch_size = 1
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
                    temperature=0.8,
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
                    match = pattern.match(first_line)
                    if match:
                        args = match.group(1)
                        expected_result = match.group(2)
                        if not args.strip() or not expected_result.strip():
                            logger.debug('Arguments or expected result is empty. Skipping.')
                            continue
                        try:
                            result_value = ast.literal_eval(expected_result)
                        except (ValueError, SyntaxError):
                            logger.debug('Expected result is not a specific value. Skipping.')
                            continue
                        try:
                            ast.parse(first_line)
                        except SyntaxError as e:
                            logger.debug(f'Syntax error in assert statement: {e}. Skipping.')
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
                        break 
                    else:
                        logger.debug('Assert statement does not match the required pattern.')

                    if attempt == 10:
                        logger.debug('Max attempts reached. Using the output from the 10th attempt.')
                        if first_line in existing_test_cases:
                            logger.debug('Duplicate test case. Skipping.')
                            continue  

                        sample = {
                            "task_id": task_id,
                            "completion": first_line,
                            "logprobs": logprobs
                        }

                        out_f.write(json.dumps(sample) + '\n')
                        out_f.flush()
                        os.fsync(out_f.fileno())  
                        existing_test_cases.add(first_line)
                        generated_count += 1
                        logger.info(f'Generated {generated_count}/{num_cases} test cases for task {task_id} (unmatched pattern)')
                        break  

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
    test_cases_output_file = f'generated_test_cases_part{process_id}.jsonl'
    for task in tasks:
        task_id = task['task_id']
        logger.info(f'Processing task {task_id}')
        generate_test_cases(task, test_cases_output_file, num_cases=100, model=model, tokenizer=tokenizer, device=device, logger=logger)

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
