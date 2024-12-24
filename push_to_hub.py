# This file no longer is needed, unless for pushing tokenizer into a new repository
# Because models are now pushed in the train script
"""
Sample from a trained model
"""
import os
import sys
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT, GPTLA, GPT_LAE, GPT_LAA

from transformers import GPT2TokenizerFast, LlamaTokenizer

# -----------------------------------------------------------------------------
out_dir = 'out' # ignored if init_from is not 'resume'
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster

model_class_name = 'GPTLA'
look_ahead_size = 2
wandb_run_path = ''
repo_id = ''
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

if not repo_id:
    raise Exception("repo_id is required.")

model_class = getattr(sys.modules[__name__], model_class_name)

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
# init from a model saved in a specific directory
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = model_class(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)


commit_message = f"WandB run: {wandb_run_path}" if wandb_run_path else None
print("pushing")
model.push_to_hub(repo_id, private=True, commit_message=commit_message)
print("pushed")


#tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer = LlamaTokenizer.from_pretrained("llama/tokenizer.model")
tokenizer.model_max_length = 512
tokenizer.push_to_hub(repo_id, private=True)
