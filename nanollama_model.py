import math
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from huggingface_hub import PyTorchModelHubMixin
from transformers import PreTrainedModel

from model import GPT, GPTConfig, Block, LayerNorm, CausalLMOutputWithCrossAttentionsAndLookAhead


def sample_top_p(probs, p):
  """
  Perform top-p (nucleus) sampling on a probability distribution.

  Args:
      probs (torch.Tensor): Probability distribution tensor.
      p (float): Probability threshold for top-p sampling.

  Returns:
      torch.Tensor: Sampled token indices.

  Note:
      Top-p sampling selects the smallest set of tokens whose cumulative probability mass
      exceeds the threshold p. The distribution is renormalized based on the selected tokens.

  """
  probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
  probs_sum = torch.cumsum(probs_sort, dim=-1)
  mask = probs_sum - probs_sort > p
  probs_sort[mask] = 0.0
  probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
  next_token = torch.multinomial(probs_sort, num_samples=1)
  next_token = torch.gather(probs_idx, -1, next_token)
  return next_token


class NanoLlamaMultiToken(GPT, PyTorchModelHubMixin, PreTrainedModel,
                          repo_url="https://github.com/HRezaei/nanoGPT",
                          pipeline_tag="text-generation",
                          license="mit", ):
  config_class = GPTConfig

  def __init__(self, config, device_map=None, **kwargs):
    super(GPT, self).__init__(config)
    config.auto_map["AutoModel"] = f"model.{self.__class__.__name__}"
    config.auto_map["AutoModelForCausalLM"] = f"model.{self.__class__.__name__}"
    assert config.vocab_size is not None
    assert config.block_size is not None
    self.config = config

    normal_layers = config.n_layer - config.look_ahead_size
    self.transformer = nn.ModuleDict(dict(
      wte=nn.Embedding(config.vocab_size, config.n_embd),
      wpe=nn.Embedding(config.block_size, config.n_embd),
      drop=nn.Dropout(config.dropout),
      h=nn.ModuleList([Block(config) for _ in range(normal_layers)]),
      ln_f=LayerNorm(config.n_embd, bias=config.bias),
    ))

    self.extra_heads = nn.ModuleDict(dict(
      wte=nn.Embedding(config.vocab_size, config.n_embd),
      wpe=nn.Embedding(config.block_size, config.n_embd),
      drop=nn.Dropout(config.dropout),
      h=nn.ModuleList([Block(config) for _ in range(self.config.look_ahead_size)]),
      ln_f=LayerNorm(config.n_embd, bias=config.bias),
    ))

    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    # with weight tying when using torch.compile() some warnings get generated:
    # "UserWarning: functional_call was passed multiple values for tied weights.
    # This behavior is deprecated and will be an error in future versions"
    # not 100% sure what this is, so far seems to be harmless. TODO investigate
    self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

    # init all weights
    self.apply(self._init_weights)
    # apply special scaled init to the residual projections, per GPT-2 paper
    for pn, p in self.named_parameters():
      if pn.endswith('c_proj.weight'):
        torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    if device_map is not None and '' in device_map:
      self.to(device_map[''])

  def forward(self, input_ids, targets=None, output_hidden_states=False, past_key_values=None, start_pos=0, **kwargs):
    if past_key_values is None:
      past_key_values = tuple([None] * (len(self.transformer.h) + len(self.extra_heads.h)))

    device = input_ids.device
    b, t = input_ids.size()
    assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
    pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

    # forward the GPT model itself
    tok_emb = self.transformer.wte(input_ids)  # token embeddings of shape (b, t, n_embd)
    pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
    x = self.transformer.drop(tok_emb + pos_emb)
    presents = ()
    all_hidden_states = ()
    head_count = self.config.look_ahead_size + 1
    past_key_values_normal = past_key_values[:-head_count]
    past_key_values_extra = past_key_values[-head_count:]
    for (block, layer_past) in zip(self.transformer.h[:-1], past_key_values_normal):
      if output_hidden_states:
        all_hidden_states = all_hidden_states + (x,)
      (x, present) = block(x, layer_past=layer_past)
      presents = presents + (present,)
    x = self.transformer.ln_f(x)
    h_trunk = x.detach()

    latents = []
    lm_heads = [self.transformer.h[-1]] + list(self.extra_heads.h)
    for (block, layer_past) in zip(lm_heads, past_key_values_extra):
      if output_hidden_states:
        all_hidden_states = all_hidden_states + (h_trunk,)
      (x, present) = block(h_trunk, layer_past=layer_past)
      presents = presents + (present,)
      latents.append(x)

    x = torch.stack(latents, dim=1)  # (_bsz, n_heads_to_use, seqlen, dim)
    x = self.extra_heads.ln_f(x)

    individual_losses = []
    # if we are given some desired targets also calculate the loss
    if targets is not None:
      if self.training and self.config.llama_loss_mode == 'original':
        head_logits = []
        head_losses = []
        for i in range(head_count):
          logits = self.lm_head(x[:, [i]])
          loss = self.compute_loss(logits, targets[:, i].contiguous())
          head_losses.append(loss)
          head_logits.append(logits)
        loss = head_losses
        logits = torch.cat(head_logits, dim=1)
      elif (not self.training) or self.config.llama_loss_mode == 'one_go':
        logits = self.lm_head(x)
        loss = self.compute_loss(logits, targets)

      # For this model, we don't differentiate between loss on input tokens and next token
      individual_losses = [
        self.compute_loss(logits[:, 0, :-1].contiguous(), targets[:, 0, :-1].contiguous())  # Input loss
      ]
      # So we only will have one loss for original head and one loss per lookahead heads:
      for i in range(self.config.look_ahead_size + 1):
        individual_losses.append(
          self.compute_loss(logits[:, i, -1].contiguous(), targets[:, i, -1].contiguous())
        )
      #individual_losses['loss'] = loss
      #individual_losses['input'] = self.compute_loss(logits[:, :-1, 0].contiguous(), targets[:, 0, :-1].contiguous()).item()
      #individual_losses['next_token'] = self.compute_loss(logits[:, [-1], 0], targets[:, 0, [-1]]).item()
    else:
      logits = self.lm_head(x)
      loss = None

    output = CausalLMOutputWithCrossAttentionsAndLookAhead(
      loss=loss,
      logits=logits[:, 0],
      past_key_values=past_key_values,
      hidden_states=all_hidden_states,  # For now, I don't need this
      attentions=None,  # For now, I don't need this
      cross_attentions=None,  # For now, I don't need this
      look_ahead_logits=logits[:, 1:],
    )
    output["individual_losses"] = torch.stack(individual_losses) if len(individual_losses) > 0 else None
    return output

  @torch.inference_mode()
  def generate_based_on_llama_source(
      self,
      prompt_tokens: List[List[int]],
      max_gen_len: int,
      temperature: float = 0.6,
      top_p: float = 0.9,
      logprobs: bool = False,
      echo: bool = False,
      top_k: int = None
  ) -> List[List[int]] :  # Tuple[List[List[int]], Optional[List[List[float]]]]:
    """
    Generate text sequences based on provided prompts using the language generation model.

    Args:
        prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
        max_gen_len (int): Maximum length of the generated text sequence.
        temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
        top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
        logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
        echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

    Returns:
        Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

    Note:
        This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
        If logprobs is True, token log probabilities are computed for each generated token.

    """
    params = self.config
    max_batch_size = 32  # For now
    device = self.device
    tokenizer_eos_id = 0  # Todo: fix this
    bsz = len(prompt_tokens)
    assert bsz <= max_batch_size, (bsz, max_batch_size)

    min_prompt_len = min(len(t) for t in prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)
    assert max_prompt_len <= params.block_size
    total_len = min(params.block_size, max_gen_len + max_prompt_len)

    pad_id = 0  # Todo: fix this. It was initially: self.tokenizer.pad_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
    for k, t in enumerate(prompt_tokens):
      tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)
    if logprobs:
      token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

    prev_pos = 0
    eos_reached = torch.tensor([False] * bsz, device=device)
    input_text_mask = tokens != pad_id
    if min_prompt_len == total_len:
      logits = self.forward(tokens, prev_pos).squeeze(2)
      token_logprobs = -F.cross_entropy(
        input=logits.transpose(1, -1),
        target=tokens.flatten,
        reduction="none",
        ignore_index=pad_id,
      )

    for cur_pos in range(min_prompt_len, total_len):
      forward_outcome = self.forward(tokens[:, prev_pos:cur_pos], start_pos=prev_pos)
      logits = forward_outcome.logits.squeeze(2)
      if temperature > 0:
        probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
      else:
        next_token = torch.argmax(logits[:, -1], dim=-1)

      next_token = next_token.reshape(-1)
      # only replace token if prompt has already been generated
      next_token = torch.where(
        input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
      )
      tokens[:, cur_pos] = next_token
      if logprobs:
        token_logprobs[:, prev_pos + 1: cur_pos + 1] = -F.cross_entropy(
          input=logits.transpose(1, -1),
          target=tokens[:, prev_pos + 1: cur_pos + 1],
          reduction="none",
          ignore_index=pad_id,
        )
      eos_reached |= (~input_text_mask[:, cur_pos]) & (
          next_token == tokenizer_eos_id
      )
      prev_pos = cur_pos
      if all(eos_reached):
        break

    if logprobs:
      token_logprobs = token_logprobs.tolist()
    out_tokens, out_logprobs = [], []
    for i, toks in enumerate(tokens.tolist()):
      # cut to max gen len
      start = 0 if echo else len(prompt_tokens[i])
      toks = toks[start: len(prompt_tokens[i]) + max_gen_len]
      probs = None
      if logprobs:
        probs = token_logprobs[i][start: len(prompt_tokens[i]) + max_gen_len]
      # cut to eos tok if any
      if tokenizer_eos_id in toks:
        eos_idx = toks.index(tokenizer_eos_id)
        toks = toks[:eos_idx]
        probs = probs[:eos_idx] if logprobs else None
      out_tokens.append(toks)
      #out_logprobs.append(probs)
    return torch.tensor(out_tokens)  # (out_tokens, out_logprobs if logprobs else None)
