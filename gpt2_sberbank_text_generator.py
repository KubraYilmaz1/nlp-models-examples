from collections import Counter

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

gpu = torch.device('cuda')
tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")
model = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2").to(device=gpu)

text = "у меня есть две собаки"
tokens = tokenizer.encode(text)
context = torch.tensor([tokens], device=gpu)
past_values = None

length = 200
repetition_penalty = 0.3
temperature = 1
threshold = 0.3
n = 3

with torch.no_grad():
    for _ in range(length):
        logits, past_key_values = model(context, past_key_values=past_values, return_dict=False)
        ngrams = zip(*[tokens[i:] for i in range(n)])
        last_ngram_count = Counter(ngrams).get(tuple(tokens[-n:]))
        softmax_temp = 1
        if last_ngram_count > 1:
            softmax_temp = (1 + repetition_penalty) ** (last_ngram_count - 1)
        effective_temp = temperature * softmax_temp
        if past_values is None:
            next_token_logits = logits[0, -1, :] / effective_temp
        else:
            next_token_logits = logits[0, :] / effective_temp
        sorted_probs, sorted_indices = torch.sort(torch.softmax(next_token_logits, dim=-1), descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        new_idx = torch.sum(cumulative_probs < threshold) + 1
        sorted_probs, sorted_indices = sorted_probs[:new_idx], sorted_indices[:new_idx]
        sorted_probs /= torch.sum(sorted_probs)
        res = sorted_indices[torch.multinomial(sorted_probs, num_samples=1).item()]
        tokens += [res.item()]
        context = res.unsqueeze(0)
        past_values = past_key_values

print(tokenizer.decode(tokens))
