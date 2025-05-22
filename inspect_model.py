# https://www.deepspeed.ai/tutorials/flops-profiler/

import os
import torch
from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import argparse
import time


def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        help="Directory name of target model. Should be located at this directory.",
        required=True,
    )
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--seq_len", type=int)

    args = parser.parse_args()

    return args


def bert_input_constructor(batch_size, seq_len, tokenizer):
    fake_seq = ""
    for _ in range(seq_len - 2):  # ignore the two special tokens [CLS] and [SEP]
        fake_seq += tokenizer.pad_token
    inputs = tokenizer(
        [fake_seq] * batch_size, padding=True, truncation=True, return_tensors="pt"
    )
    labels = torch.tensor([1] * batch_size)
    inputs = dict(inputs)
    inputs.update({"labels": labels})
    return inputs


def get_pt_path(model_path):
    if not os.path.isdir(model_path):
        return None

    for filename in os.listdir(model_path):
        file_extension = os.path.splitext(filename)[1]
        if file_extension == ".pt":
            pt_path = os.path.join(model_path, filename)
            print(f"Found PyTorch model `{pt_path}'")
            return pt_path
        
    return None


def inspect(model, batch_size, seq_len):
    with get_accelerator().device(0):
        tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=True,
            truncation_side="left",
            padding_side="right",
        )

        pt_path = get_pt_path(model)
        if pt_path:
            model = torch.load(pt_path, weights_only=False)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model,
            )

        # https://stackoverflow.com/a/73137031
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))

        inputs = bert_input_constructor(batch_size, seq_len, tokenizer)["input_ids"]

        flops, macs, params = get_model_profile(
            model,
            args=[inputs],
            print_profile=True,
            detailed=True,
        )


def main():
    start_time = time.time()

    args = parse_arg()

    inspect(args.model, args.batch_size, args.seq_len)

    end_time = time.time()
    elapsed_time_s = end_time - start_time
    elapsed_time_m = elapsed_time_s / 60
    elapsed_time_h = elapsed_time_m / 60
    elapsed_str = f"Total Running Time : {(elapsed_time_s):.2f} sec = {(elapsed_time_m):.2f} min = {(elapsed_time_h):.2f} hr"
    print(f"\n{elapsed_str}")


if __name__ == "__main__":
    main()
