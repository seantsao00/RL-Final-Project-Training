import argparse
import tqdm
import torch
import jsonlines
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

"""
git clone https://github.com/openai/human-eval
$ pip install -e human-eval
evaluate_functional_correctness sample-output-file
"""


def decode(tokens_list, tokenizer, raw_text_len):
    sents = []
    # print(len(tokens_list))
    for tokens in tokens_list:
        tokens = tokens.cpu().numpy().tolist()
        sent = tokenizer.decode(tokens[raw_text_len:])
        sent = sent.split("<|endoftext|>")[0]
        sent = sent.split("\n\n\n")[0]
        sent = sent.split("\n\n")[0]
        sent = sent.split("def ")[0]
        sents.append(sent)
    return sents


def generate_sample(model, tokenizer, input_txt):
    # Tokenize with attention mask
    inputs = tokenizer(
        input_txt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)
    
    raw_text_len = inputs.input_ids.shape[1]
    
    print(f"Input text: {input_txt}\n")
    
    # Generate with proper parameters
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
    )
    
    output_text = decode(outputs, tokenizer, raw_text_len)[0]
    print(f"\nOutput text: \n{output_text}\n")
    return output_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        help="Checkpoint path",
        default="Qwen/Qwen-7B",
    )
    parser.add_argument(
        "-f",
        "--sample-input-file",
        type=str,
        default=None,
        help="data path to HumanEval.jsonl",
    )
    parser.add_argument(
        "-o", "--sample-output-file", type=str, default="HumanEval_res.jsonl"
    )

    args = parser.parse_args()
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )
    
    # Set pad_token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.eos_token}")

    print("Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path, 
        device_map="auto", 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).eval()
    
    # Set generation config
    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )
    model.generation_config.do_sample = False
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    f_output = jsonlines.Writer(open(args.sample_output_file, "w", encoding="utf-8"))

    f = jsonlines.open(args.sample_input_file)
    with f_output as output:
        for jobj in tqdm.tqdm(f, desc="task_idx"):
            prompt = jobj["prompt"]
            task_id = jobj["task_id"]
            gen_sents = generate_sample(model, tokenizer, prompt)
            gen_jobjs = {"task_id": task_id, "completion": gen_sents}
            output.write(gen_jobjs)
    f_output.close()
