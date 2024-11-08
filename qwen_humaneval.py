from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
output_logits = True
repeat = 100

torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define prompt template
prompt_template = """
                Complete the function below to solve the coding problem: \n
                ------------------------------------------------------------- \n
                {code_prompt}\
                """

def write_json_file(dicts, output_file):
    with open(output_file, 'w') as f:
        for d in dicts:
            f.write(json.dumps(d)+'\n')

def process_model_input(prompt):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    return model_inputs


# init model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto"
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# init dataset
dataset = load_dataset("openai_humaneval")



outputs = []
for i, data in enumerate(dataset['test']):
    print(f"{i} / {len(dataset['test'])}")
    # parse data
    task_id = data['task_id']
    prompt = data['prompt']
    

    # repeat 100 times
    for r in tqdm(repeat):
        # build model input
        prompt = prompt_template.format(code_prompt=prompt)
        model_inputs = process_model_input(prompt)

        # process output
        generation_output = model.generate(
            **model_inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
        generated_ids = generation_output.sequences
        logprobs = model.compute_transition_scores(generation_output.sequences, generation_output.scores, normalize_logits=True)
        logprobs = logprobs[0].detach().cpu().tolist()
        #logits = generation_output.sequences.logits
        #logprobs = F.log_softmax(logits, dim=-1).detach().cpu().to_list()

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # record output
        output_dict = {"task_id": task_id, \
                        "completion": response, \
                        "logprobs": logprobs, \
                        }
        outputs.append(output_dict) 
    

# write to output file
write_json_file(outputs, output_file)
