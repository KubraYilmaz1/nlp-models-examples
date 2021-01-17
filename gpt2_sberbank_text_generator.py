import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

gpu = torch.device('cuda')
tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")
model = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2").to(device=gpu)

text = "у меня есть две собаки"
tokens = tokenizer.encode(text)

length = 150
repetition_penalty = 1.5
temperature = 1

for _ in range(length):
    tokens_tensor = torch.tensor([tokens], device=gpu)
    logits = model(tokens_tensor)[0]
    next_token_logits = logits[0, -1, :] / temperature
    for token in set(tokens):
        next_token_logits[token] /= repetition_penalty
    next_token = torch.argmax(next_token_logits).item()
    tokens += [next_token]

print(tokenizer.decode(tokens))
