from transformers import AutoModel, AutoConfig, AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer, GPT2TokenizerFast



model_path = "hrezaei/nanoGPT"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, private=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, private=True)
print(model)
print(model.config)
print(tokenizer)
