from transformers import AutoModel, AutoConfig, AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer, GPT2TokenizerFast

from model import GPTConfig, GPTLA

AutoConfig.register("nanogpt", GPTConfig)
AutoTokenizer.register(GPTConfig, GPT2Tokenizer, GPT2TokenizerFast)
AutoModel.register(GPTConfig, GPTLA)
AutoModelForCausalLM.register(GPTConfig, GPTLA)


model_path = "hrezaei/nanoGPTLookAhead"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, private=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, private=True)
print(model)
print(model.config)
print(tokenizer)
