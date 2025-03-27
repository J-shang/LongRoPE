# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import sys
import logging
import argparse
import random
import numpy as np

import datasets
from transformers import AutoTokenizer, PreTrainedTokenizer


logger = logging.getLogger(__file__)


def find_closest_index(nums, target):
    nums_array = np.array(nums)
    index = (np.abs(nums_array - target)).argmin()
    return index


def insert_needle(tokenizer: PreTrainedTokenizer, text: str, needle: str, insert_pos_ratio: float, target_length: int):
    tokenized_ids = tokenizer.encode(text, add_special_tokens=False)
    break_token_ids = [tokenizer.encode('\n', add_special_tokens=False)[0], tokenizer.encode(',\n', add_special_tokens=False)[0], tokenizer.encode('.\n', add_special_tokens=False)[0]]
    break_list = [_ for _, num in enumerate(tokenized_ids) if num in break_token_ids]

    t1 = f"A special magic number is hidden within the following text. Make sure to memorize it. I will quiz you about the number afterwards.\n"
    t1 = tokenizer.encode(t1, add_special_tokens=False)
    t2 = f"One of the special magic numbers for wandering-age is: {needle}.\n"
    t2 = tokenizer.encode(t2, add_special_tokens=False)
    t3 = f"The special magic number for wandering-age mentioned in the provided text is: {needle}"
    t3 = tokenizer.encode(t3, add_special_tokens=False)

    text_length = target_length - len(t1) - len(t2) - len(t3) - 1
    insert_pos = int(insert_pos_ratio * text_length)
    closest_break_idx = find_closest_index(break_list, insert_pos)
    break_idx = break_list[closest_break_idx] if break_list[closest_break_idx] < text_length else break_list[closest_break_idx-1]

    front, back = tokenized_ids[:break_idx+1], tokenized_ids[break_idx+1:text_length]

    needle_ids = tokenizer.encode(str(needle), add_special_tokens=False)
    input_ids = [tokenizer.bos_token_id] + t1 + front + t2 + back + t3
    cus_label = [-100] * (len(input_ids) - len(needle_ids)) + needle_ids
    assert len(input_ids) == len(cus_label)
    return input_ids, cus_label


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    input_texts = datasets.load_dataset(
        args.dataset,
        name=args.subset,
        split=args.split,
        num_proc=args.num_proc,
        ignore_verifications=True,
        trust_remote_code=True,
    )
    input_texts.filter(lambda x: args.target_length < len(x[args.feature]) < 2 * args.target_length)

    def tokenize(examples: list):
        insert_pos_ratios = [_ / len(examples) for _ in range(len(examples))]
        insert_pos_ratio_gap = insert_pos_ratios[1] - insert_pos_ratios[0]
        insert_pos_ratios = [_ + insert_pos_ratio_gap * random.random() for _ in insert_pos_ratios]
        needles = [str(random.randint(1000000, 9999999)) for _ in range(len(examples))]
        res = []
        for text, needle in zip(examples, needles):
            input_ids, cus_label = insert_needle(tokenizer, text, needle, args.target_length)
            res.append({
                "input_ids": input_ids,
                "cus_label": cus_label,
                "attention_mask": [1] * len(input_ids),
                "tokenized_len": len(input_ids),
            })
        return res

    input_texts = input_texts.map(tokenize, num_proc=args.num_proc, batched=True, batch_size=10)
    input_texts.save_to_disk(args.save_tokenized, num_proc=args.num_proc)
    logger.info(f"Saved tokenized dataset to {args.save_tokenized}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--subset", type=str)
    parser.add_argument("--feature", type=str)
    parser.add_argument("--save-tokenized", type=str)
    parser.add_argument("--num-proc", type=int, default=4)
    parser.add_argument("--target-length", type=int, default=131072)

    main(parser.parse_args())
