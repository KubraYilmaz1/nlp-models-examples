import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

gpu = torch.device('cuda')
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2-xl", pad_token_id=tokenizer.eos_token_id).to(device=gpu)

text = "I have two dogs and one is a"
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

# "I have two dogs and one is a pit bull. I don't want to be around them." The dog owner, who did not wish to give
# his name for fear of reprisals from the animal rights group that has been campaigning against him since he adopted
# it in 2010, said: "It's very upsetting because they are my family members but we can never go back there now. We
# will always live with this guilt over what happened here."
#
# A new study by researchers at Harvard University suggests that people may actually prefer their own gender when
# making decisions about romantic partners — even if those choices aren't based on biological sex or sexual
# orientation. The findings suggest that our preferences might change as society becomes more accepting of same-sex
# relationships.
print(tokenizer.decode(tokens))
