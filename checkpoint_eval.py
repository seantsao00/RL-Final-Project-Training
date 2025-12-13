#!/usr/bin/env python3
import argparse
import os
import tqdm
import torch
import jsonlines
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from peft import PeftModel


def decode(tokens_list, tokenizer, raw_text_len):
    sents = []
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
    inputs = tokenizer(
        input_txt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)
    raw_text_len = inputs.input_ids.shape[1]
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=1024,
    )
    output_text = decode(outputs, tokenizer, raw_text_len)[0]
    return output_text


def main():
    parser = argparse.ArgumentParser(description="Evaluate a single checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="human-eval/data/HumanEval.jsonl"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True
    )
    parser.add_argument(
        "--no-lora",
        action="store_true"
    )
    parser.add_argument(
        "--print-all",
        action="store_true"
    )
    
    args = parser.parse_args()
    print("="*80)
    print("GENERATING COMPLETIONS")
    print("="*80)
    print("\nLoading tokenizer...")
    model_path = args.checkpoint if args.no_lora else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.eos_token}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if not args.no_lora:
        model = PeftModel.from_pretrained(model, args.checkpoint)
    
    model.eval()
    model.generation_config = GenerationConfig(
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(f"Output will be saved to: {args.output}\n")
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    f_output = jsonlines.Writer(open(args.output, "w", encoding="utf-8"))
    f = jsonlines.open(args.input_file)
    results_summary = []
    
    with f_output as output:
        for idx, jobj in enumerate(tqdm.tqdm(f, desc="Generating"), 1):
            prompt = jobj["prompt"]
            task_id = jobj["task_id"]
            gen_sents = generate_sample(model, tokenizer, prompt)
            gen_jobjs = {"task_id": task_id, "completion": gen_sents}
            output.write(gen_jobjs)
            results_summary.append({
                "task_id": task_id,
                "completion": gen_sents
            })
            if args.print_all or idx % 10 == 0 or idx <= 3:
                print(f"\n{'='*80}")
                print(f"Task {idx}: {task_id}")
                print(f"{'='*80}")
                print(f"Prompt:\n{prompt[:200]}{'...' if len(prompt) > 200 else ''}")
                print(f"\nGenerated Completion:\n{gen_sents}")
    
    f_output.close()
    
    print(f"\n{'='*80}")
    print("GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Results saved to: {args.output}")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

