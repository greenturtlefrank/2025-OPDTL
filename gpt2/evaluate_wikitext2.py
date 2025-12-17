import torch
import math
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
from tqdm import tqdm
from model import GPTConfig, GPT
import tiktoken
import matplotlib.pyplot as plt


def calculate_perplexity(model, encodings, attn_topk_ratio, ffn_topk_ratio):
    seq_len = len(encodings)

    # Define the sliding window parameters
    max_length = 1024  # 1024 for GPT-2
    stride = 512  # Amount to slide the window

    total_nll = 0.0  # Total negative log-likelihood
    total_tokens = 0 # Total number of tokens evaluated

    with torch.no_grad():
        # Use tqdm for a progress bar
        for i in tqdm(range(0, seq_len, stride)):
            # Define the start and end of the current window
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, seq_len)

            # Get the input chunk (length = max_length, except for the last one)
            input_ids = encodings[begin_loc:end_loc]
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)[None, ...]

            # Create labels, which are the same as input_ids for language modeling
            labels = input_ids.clone()
            
            # --- Crucial Step: Masking Overlapping Tokens ---
            # We only want to calculate the loss on the new 'stride' tokens,
            # not the overlapping 'context' tokens.
            
            if i > 0:
                # The length of the overlapping context
                context_len = max_length - stride 
                if context_len > 0:
                    labels[:, :context_len] = -1        


            # --- Forward Pass ---
            outputs, loss = model(input_ids[:, :-1], labels[:, 1:], attn_topk_ratio, ffn_topk_ratio)

            # outputs.loss is the *average* NLL for the *unmasked* tokens
            # We need the *total* NLL, so we multiply by the number of tokens
            # that were actually evaluated.
            
            # Count the number of unmasked tokens
            num_unmasked_tokens = (labels != -1).sum().item()
            
            if num_unmasked_tokens > 0:
                # Total NLL for this chunk = mean_NLL * num_tokens
                chunk_total_nll = loss.item() * num_unmasked_tokens
                
                # Accumulate
                total_nll += chunk_total_nll
                total_tokens += num_unmasked_tokens
    # Average NLL per token
    avg_nll = total_nll / total_tokens
    perplexity = math.exp(avg_nll)

    return perplexity
# 0. Load the model
# -----------------------------------------------------------------------------
# You can change this to 'gpt2-medium', 'gpt2-large', or 'gpt2-xl'
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

# Use the 'wikitext' dataset, 'wikitext-2-raw-v1' configuration
dataset_name = "wikitext"
dataset_config = "wikitext-2-raw-v1"
dataset_split = "test" # Standard evaluation split

# --- 2. Load Tokenizer ---
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# --- 3. Load and Prepare Dataset ---
print(f"Loading dataset: {dataset_name} ({dataset_config})")
# We use the 'test' split for evaluation
dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
# WikiText-2 is split by lines/paragraphs. 
# Filter out empty lines, then join all text with the EOT token.
eot_token = "<|endoftext|>"
non_empty_lines = [text for text in dataset['text'] if text.strip()]
all_text = " ".join(non_empty_lines)

# Tokenize the entire dataset
encodings = encode(all_text)


# --- 4. Calculate Perplexity with Sliding Window ---
topk_ratios = [round(i*0.1, 2) for i in range(1, 11, 1)]
perplexities = []
for topk in topk_ratios:
    print(f"Evaluating perplexity over top_k ratio {topk}...")
    perplexity = calculate_perplexity(model, encodings, topk, 1)
    perplexities.append(perplexity)
    print(f"Perplexity at top_k ratio {topk}: {perplexity:.4f}")


out_path = "results/wikitext_gpt2-xl_attn.png"
plt.figure(figsize=(8,6))
plt.rcParams['font.size'] = 16
plt.plot(topk_ratios, perplexities, marker='o')
plt.title('contextual sparisty of attention layer of gpt2-xl on wikitext2')
plt.xlabel('density')
plt.ylabel('perplexity')
plt.savefig(out_path, bbox_inches='tight')
print(f"Saved plot to: {out_path}")

