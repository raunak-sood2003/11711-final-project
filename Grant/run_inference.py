import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import multiprocessing as mp
from tqdm import tqdm
import logging

def split_tasks(lines, num_splits):
    tasks = [json.loads(line) for line in lines]
    return [tasks[i::num_splits] for i in range(num_splits)]

def run_inference_on_gpu(gpu_id, task_subset, output_file, batch_size=4):
    import os
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from tqdm import tqdm
    import logging
    import math


    logging.basicConfig(
        filename=f'gpu_{gpu_id}_log.txt',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(message)s'
    )
    logger = logging.getLogger()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        logger.info(f'GPU {gpu_id}: Setting device')
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')

        logger.info(f'GPU {gpu_id}: Loading tokenizer')
        tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/deepseek-coder-6.7b-base",
            trust_remote_code=True
        )

        logger.info(f'GPU {gpu_id}: Loading model')
        model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/deepseek-coder-6.7b-base",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to(device)

        logger.info(f'GPU {gpu_id}: Model loaded')

        with open(output_file, 'w') as out_f:
            for task in tqdm(task_subset, desc=f"GPU {gpu_id}", position=gpu_id):
                prompt = task['prompt']
                task_id = task['task_id']

                num_completions = 100
                for start_idx in range(0, num_completions, batch_size):
                    current_batch_size = min(batch_size, num_completions - start_idx)
                    prompts = [prompt] * current_batch_size
                    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
                    input_ids = inputs['input_ids']
                    attention_mask = inputs['attention_mask']

                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=128,
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

                        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

                        sample = {
                            "task_id": task_id,
                            "completion": generated_text,
                            "logprobs": logprobs
                        }

                        out_f.write(json.dumps(sample) + '\n')

        logger.info(f'GPU {gpu_id}: Inference completed')

    except Exception as e:
        logger.exception(f"Exception on GPU {gpu_id}")
        print(f"Exception on GPU {gpu_id}: {e}")

def combine_output_files(num_gpus, output_filename):
    with open(f'dataGrant/{output_filename}', 'w') as outfile:
        for gpu_id in range(num_gpus):
            input_file = f'dataGrant/generated_samples_gpu{gpu_id}.jsonl'
            with open(input_file, 'r') as infile:
                outfile.write(infile.read())

def main():
    if not os.path.exists('dataGrant'):
        os.makedirs('dataGrant')

    humaneval_path = "data/humaneval-python.jsonl"

    with open(humaneval_path, 'r') as f:
        lines = f.readlines()

    num_gpus = 8
    task_subsets = split_tasks(lines, num_gpus)
    mp.set_start_method('spawn')
    processes = []
    for gpu_id in range(num_gpus):
        task_subset = task_subsets[gpu_id]
        output_file = f'dataGrant/generated_samples_gpu{gpu_id}.jsonl'
        p = mp.Process(target=run_inference_on_gpu, args=(gpu_id, task_subset, output_file))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    combine_output_files(num_gpus, 'generated_samples.jsonl')

if __name__ == "__main__":
    main()