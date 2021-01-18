from collections import Counter

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

gpu = torch.device('cuda')
tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")
model = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2").to(device=gpu)

text = "у меня есть две собаки"
tokens = tokenizer.encode(text)

length = 200
repetition_penalty = 0.1
temperature = 1
threshold = 0.3
n = 3

with torch.no_grad():
    for _ in range(length):
        logits = model(torch.tensor([tokens], device=gpu))[0]
        ngrams = zip(*[tokens[i:] for i in range(n)])
        last_ngram_count = Counter(ngrams).get(tuple(tokens[-n:]))
        softmax_temp = 1
        if last_ngram_count > 1:
            softmax_temp = (1 + repetition_penalty) ** (last_ngram_count - 1)
        effective_temp = temperature * softmax_temp
        next_token_logits = logits[0, -1, :] / effective_temp
        sorted_probs, sorted_indices = torch.sort(torch.softmax(next_token_logits, dim=-1), descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        new_idx = torch.sum(cumulative_probs < threshold) + 1
        sorted_probs, sorted_indices = sorted_probs[:new_idx], sorted_indices[:new_idx]
        sorted_probs /= torch.sum(sorted_probs)
        res = sorted_indices[torch.multinomial(sorted_probs, num_samples=1).item()]
        tokens += [res.item()]

print(tokenizer.decode(tokens))
