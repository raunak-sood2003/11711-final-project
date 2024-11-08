from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import math
#import torch

model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
output_logits = True
output_file = "qwen_humaneval_programs_k100_temp0.8.jsonl"
repeat = 100
batch_size = 4
mode = "program"

#torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define prompt template
if mode == "program":
    prompt_template = """
                    Just complete the function below and do not explain: \n
                    {code_prompt}\
                    """
elif mode == "test":
    prompt_template = """
                    Generate test cases for the function below and do not explain: \n
                    {code_prompt}\
                    """
else:
    raise Exception("mode should be either program or test")


def write_json_file(dicts, output_file):
    with open(output_file, 'w') as f:
        for d in dicts:
            f.write(json.dumps(d)+'\n')

def process_model_input(prompt, batch_size):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text]*batch_size, return_tensors="pt").to(model.device)
    return model_inputs

def fix_infinity(logprobs):
    for i in range(len(logprobs)):
        if math.isinf(logprobs[i]):
            logprobs[i] = -1e10
    return logprobs


# init model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto", 
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# init dataset
dataset = load_dataset("openai_humaneval")



outputs = []
for i, data in tqdm(enumerate(dataset['test'])):
    print(f"{i} / {len(dataset['test'])}")
    # parse data
    task_id = data['task_id']
    prompt = data['prompt']
    

    # repeat 100 times
    for r in range(0, repeat, batch_size):
        # build model input
        prompt = prompt_template.format(code_prompt=prompt)
        model_inputs = process_model_input(prompt, batch_size)
        input_length = model_inputs['input_ids'].size(1)

        # process output
        generation_output = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
        logprobs = model.compute_transition_scores(generation_output.sequences, generation_output.scores, normalize_logits=True)
        generated_ids = generation_output.sequences[:, input_length:]
        assert generated_ids.size(1) == logprobs.size(1)
        
        # decoding
        for j in range(batch_size):
            logprobs_sample = logprobs[j].detach().cpu().tolist()
            logprobs_sample = fix_infinity(logprobs_sample)
            response = tokenizer.decode(generated_ids[j], skip_special_tokens=True)

            # record output
            output_dict = {"task_id": task_id, \
                            "completion": response, \
                            "logprobs": logprobs_sample, \
                            }
            #breakpoint()
            outputs.append(output_dict) 
    

# write to output file
write_json_file(outputs, output_file)
