import json

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer, GPT2TokenizerFast

from gptla_model.configuration_gptla import GPTConfig
from gptla_model.modeling_gptla import GPTLA

AutoConfig.register("nanogpt", GPTConfig)
AutoTokenizer.register(GPTConfig, GPT2Tokenizer, GPT2TokenizerFast)
AutoModel.register(GPTConfig, GPTLA)
AutoModelForCausalLM.register(GPTConfig, GPTLA)

GPTConfig.register_for_auto_class()
GPTLA.register_for_auto_class("AutoModel")
GPTLA.register_for_auto_class("AutoModelForCausalLM")

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
