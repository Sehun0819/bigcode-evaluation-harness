# https://www.deepspeed.ai/tutorials/flops-profiler/

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
    parser.add_argument(
        "--pt",
        type=str,
        help="(Our custom models only) File name of PyTorch model (.py). Should be located at model directory.",
        default="",
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


def inspect(model, pt, batch_size, seq_len):
    with get_accelerator().device(0):
        tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=True,
            truncation_side="left",
            padding_side="right",
        )

        if pt:
            model = torch.load(pt, weights_only=False)
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

    inspect(args.model, args.pt, args.batch_size, args.seq_len)

    end_time = time.time()
    elapsed_time_s = end_time - start_time
    elapsed_time_m = elapsed_time_s / 60
    elapsed_time_h = elapsed_time_m / 60
    elapsed_str = f"Total Running Time : {(elapsed_time_s):.2f} sec = {(elapsed_time_m):.2f} min = {(elapsed_time_h):.2f} hr"
    print(f"\n{elapsed_str}")


if __name__ == "__main__":
    main()
