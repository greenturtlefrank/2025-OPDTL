from datasets import load_dataset
from model import GPTConfig, GPT
import torch
import tiktoken
from tqdm import tqdm
import re
import matplotlib.pyplot as plt

def normalize_text(text):
    """
    Simple normalizer to handle spacing and punctuation for comparison.
    Strips leading/trailing whitespace and lowercases.
    """
    return text.strip().lower()

def evaluate_accuracy(dataset, model, encode, decode,  attn_topk_ratio, ffn_topk_ratio):
    temperature = 1.0
    top_k = 1
    correct = 0
    total = len(dataset)

    for row in tqdm(dataset):
        full_text = row["text"]

        # Extract context (everything except the last word) and target
        # Split text into context and target (last word)
        # LAMBADA is designed such that the last word is the target.
        # We split by the last whitespace.
        full_text_clean = full_text.strip()
        last_space_index = full_text_clean.rfind(' ')

        if last_space_index == -1:
            # Skip single-word lines if any
            continue
            
        context = full_text_clean[:last_space_index]
        target_word = full_text_clean[last_space_index+1:]

        
        # Tokenize context
        input_ids = encode(context)
        inputs = (torch.tensor(input_ids, dtype=torch.long, device=device)[None, ...])

        block_size = 1024
        inputs_cond = inputs if inputs.size(1) <= block_size else inputs[:, -block_size:]

        # Generate the next tokens greedily (up to 10 new tokens to handle subwords)
        with torch.no_grad():
            with ctx:
                outputs = model.generate(inputs_cond, max_new_tokens, temperature=temperature, top_k=top_k, attn_topk_ratio=attn_topk_ratio, ffn_topk_ratio=ffn_topk_ratio)
        
        # Decode only the newly generated tokens
        # print(context)
        # print("+========================++")
        # print(decode(outputs[0].tolist()))
        # print("+===================+")
        generated_text = decode(outputs[0][inputs_cond.shape[1]:].tolist())
        
        predicted_word = generated_text.strip().split(' ')[0]

        # Clean target for comparison (remove potential punctuation attached in raw text)
        # Note: In strict evaluation, you might want exact string matching. 
        # Here we normalize to be robust against punctuation nuances.
        target_clean = re.sub(r'[^\w\s]', '', target_word)
        pred_clean = re.sub(r'[^\w\s]', '', predicted_word)

        if normalize_text(pred_clean) == normalize_text(target_clean):
            correct += 1


    accuracy = correct / total * 100
    return accuracy

# 0. Load the model
# -----------------------------------------------------------------------------
model_id = "gpt2-xl"
num_samples = 1 # number of samples to draw
max_new_tokens = 4 # number of tokens generated in each sample
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 1 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster
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

# Load the LAMBADA dataset (test split)
# dataset = load_dataset("lambada_openai", "default")["test"]
dataset = load_dataset("lambada", split="test")
# examples = []
# with open("lambada_test_plain_text.txt", "r") as f:
#     for line in f.readlines():
#         examples.append(line)


topk_ratios = [round(i*0.1, 2) for i in range(1, 11, 1)]
accuracies = []
for topk_ratio in topk_ratios:
    print(f"Evaluating with top_k={topk_ratio}...")
    accuracy = evaluate_accuracy(dataset, model, encode, decode, topk_ratio, 1)

    print(f"Accuracy on LAMBADA: {accuracy:.2f}%")
    accuracies.append(accuracy)

# Plot and save
out_path = "results/lambada_gpt2-xl_attn.png"
plt.figure(figsize=(8,6))
plt.rcParams['font.size'] = 16
plt.plot(topk_ratios, accuracies, marker='o')
plt.title('contextual sparisty of attention of gpt2-xl on LAMBADA')
plt.xlabel('density')
plt.ylabel('accuracy')
plt.savefig(out_path, bbox_inches='tight')
print(f"Saved plot to: {out_path}")

