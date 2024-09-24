# This file no longer is needed, unless for pushing tokenizer into a new repository
# Because models are now pushed in the train script
import json

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer, GPT2TokenizerFast

from gptla_model.configuration_gptla import GPTConfig
from gptla_model.modeling_gptla import GPTLA


model_args = json.load(open("nanoGPTLookAhead/config.json", "r"))
gptconf = GPTConfig(**model_args)


model = GPTLA(gptconf)
model.from_pretrained("nanoGPTLookAhead", config=gptconf)
repo_id = "hrezaei/nanoGPTLookAhead"
print("pushing")

model.push_to_hub(repo_id, private=True)
print("push completed")
print(model)

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

tokenizer.push_to_hub(repo_id, private=True)
