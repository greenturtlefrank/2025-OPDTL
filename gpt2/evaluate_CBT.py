import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib.pyplot as plt

def parse_cbt_file(filepath):
    """
    Parses a CBT dataset file (e.g., cbtest_CN_test.txt) into a list of examples.
    Each example contains the context, query, candidates, and correct answer.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    examples = []
    i = 0
    while i < len(lines):
        context_sentences = []
        for j in range(20):
            line = lines[i + j].strip()
            if line:
                # Split the line number from the sentence
                num, sentence = line.split(' ', 1)
                context_sentences.append(sentence.strip())

        query_line = lines[i + 20].strip()
      
        if query_line:
            # The query line format: "21 query with XXXXX\tcandidates|joined|by|pipe\tcorrect_answer\t(optional field)"
            parts = query_line.split('\t')
            query_part = parts[0].split(' ', 1)[1].strip()  # Query after '21 '
            candidates = parts[3].split('|')
            correct_answer = parts[1].strip()

            examples.append({
                'context': '\n'.join(context_sentences),
                'query': query_part,
                'candidates': candidates,
                'answer': correct_answer
            })
        
        i += 22
    
    return examples

def evaluate_accuracy(examples, predictions):
    """
    Evaluates the accuracy by comparing the generated predictions to the correct answers.
    Assumptions:
    - predictions is a list of strings, each being the predicted word for the corresponding example.
    - Length of predictions must match the number of examples.
    
    Returns the accuracy as a percentage.
    """
    if len(predictions) != len(examples):
        raise ValueError("Number of predictions must match the number of examples.")
    
    correct = 0
    for ex, pred in zip(examples, predictions):
        if pred.strip().lower() == ex['answer'].lower():
            correct += 1
    
    accuracy = (correct / len(examples)) * 100 if examples else 0
    return accuracy


def get_gpt2_prediction(context, query, candidates, model, tokenizer, attn_topk_ratio, ffn_topk_ratio):
    """
    Scores each candidate by computing the log probability of the full sequence with the candidate inserted.
    Returns the highest-scoring candidate.
    """    
    # Replace 'XXXXX' with a placeholder token to build the full input
    full_inputs = []
    for cand in candidates:
        filled_query = query.replace('XXXXX', cand)
        full_text = context + '\n\n' + filled_query  # Concat context and filled query
        input_ids = tokenizer(full_text)
        inputs = torch.tensor(input_ids, dtype=torch.long, device=device)[None, ...]
        full_inputs.append((inputs, cand))

    scores = []
    with torch.no_grad():
        for inputs, cand in full_inputs:
            # self.config.block_size
            block_size = 1024
            inputs_cond = inputs if inputs.size(1) <= block_size else inputs[:, -block_size:]
            outputs, loss = model(inputs_cond[:, :-1], inputs_cond[:, 1:], topk_ratio)
            loss = loss.item()  # Negative log likelihood (higher loss = lower prob)
            scores.append(-loss)  # Use negative loss as score (higher is better)

    best_idx = scores.index(max(scores))
    return candidates[best_idx]


# 0. Load the model
# -----------------------------------------------------------------------------
model_id = "gpt2-xl"
num_samples = 1 # number of samples to draw
max_new_tokens = 4 # number of tokens generated in each sample
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

model = GPT.from_pretrained(model_id, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# load encoder
print("No meta.pkl found, assuming GPT-2 encodings...")
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)


# Example usage:
# 1. Parse the dataset
filepath = 'CBTest/data/cbtest_CN_test_2500ex.txt'  # Replace with actual path
examples = parse_cbt_file(filepath)

topk_ratios = [round(i*0.1, 2) for i in range(1, 11, 1)]
accuracies = []
for topk_ratio in topk_ratios:
    print(f"Evaluating with top_k={topk_ratio}...")

    # 2. Generate predictions using your LLM (placeholder - replace with actual LLM calls)
    predictions = []
    for ex in tqdm(examples):
        pred = get_gpt2_prediction(ex['context'], ex['query'], ex['candidates'], model, encode, topk_ratio, 1)
        predictions.append(pred)

    # 3. Compute accuracy
    acc = evaluate_accuracy(examples, predictions)
    print(f"Accuracy on CBT-CN: {acc:.2f}%")
    accuracies.append(acc)

# Plot and save
out_path = "results/cbt_cn_gpt2-xl_attn.png"
plt.figure(figsize=(8,6))
plt.rcParams['font.size'] = 16
plt.plot(topk_ratios, accuracies, marker='o')
plt.title('contextual sparisty of attention layer of gpt2-xl on CBT-CN')
plt.xlabel('density')
plt.ylabel('accuracy')
plt.savefig(out_path, bbox_inches='tight')
print(f"Saved plot to: {out_path}")
