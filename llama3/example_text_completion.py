# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List

import fire

from llama import Llama

import re
import pandas as pd
from tqdm import tqdm


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8, #0.8
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    """
    Examples to run with the pre-trained models (no fine-tuning). Prompts are
    usually in the form of an incomplete text prefix that the model can then try to complete.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.
    `max_gen_len` is needed because pre-trained models usually do not stop completions naturally.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # prompts: List[str] = [
    #     # For these prompts, the expected answer is the natural continuation of the prompt
    #     "I believe the meaning of life is",
    #     "Simply put, the theory of relativity states that ",
    #     """A brief message congratulating the team on the launch:

    #     Hi everyone,

    #     I just """,
    #     # Few shot prompt (providing a few examples before asking model to complete more);
    #     """Translate English to French:

    #     sea otter => loutre de mer
    #     peppermint => menthe poivrée
    #     plush girafe => girafe peluche
    #     cheese =>""",
    # ]
    
    correct = 0
    dataset = []
    #data_file = "mgsm/mgsm_en.tsv"
    data_file = "mmlu/high_school_mathematics/test-00000-of-00001.parquet"
    
    rows = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                dataset.append(parts)
    
    for i, data in enumerate(tqdm(dataset)):
        
        contest = data[0].strip()
        answer = data[1].strip()
        
        prompts = [contest]
        
        results = generator.text_completion(
            prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        
        for prompt, result in zip(prompts, results):
            print(prompt)
            print(f"> {result['generation']}")
            print("\n==================================\n")
            
            
            # nums = re.findall(r"\\boxed\{([-+]?\d*\.?\d+)\}", result['generation'].strip())
            # if not nums:
            #     nums = "N"
    
            # print(nums[-1], answer)
            # if nums[-1] == answer.lower().strip():
            #     correct += 1
            
            
            nums = re.findall(r"The final answer is[^-\d]*?([-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?)", result['generation'].strip(), flags=re.IGNORECASE)
            #nums = re.findall(r"\\boxed\{([-+]?\d*\.?\d+)\}", result['generation'].strip())
            if nums:
                nums_float = [float(x.replace(",", "")) for x in nums]
                print(nums_float)
                
            
            if nums and float(answer.replace(",", "")) in nums_float:
                print("Correct!!")
                correct += 1
            else:
                print("Wrong!!")
                
            print(answer)
                        
    
    print("Acc:", correct/len(dataset))


if __name__ == "__main__":
    fire.Fire(main)
