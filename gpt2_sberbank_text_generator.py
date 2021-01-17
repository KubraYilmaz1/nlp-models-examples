import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

gpu = torch.device('cuda')
tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")
model = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2").to(device=gpu)

text = "на словах ты лев толстой"
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

# на словах ты лев толстой, а на деле - толстый и ленивый. А я не люблю толстых людей! Я их боюсь... И вообще мне
# нравятся худые люди) ) А если серьезно то это просто стереотип такой что у полных женщин больше шансов найти себе
# спутника жизни чем с нормальным весом))))) Но все же есть исключения из правил)))) Удачи тебе в поиске своего
# идеала!!!!!!! Счастья!!!: ** :-** :) У меня рост 165 см вес 55 кг. Мне кажется,что мой идеальный мужчина должен
# быть высоким (170см), стройным(180кг). Ну или хотя бы немного полноватым.. Хотя нет ни одного мужчины который был б
# со мной согласен....  Вот так вот...:) Всем удачи!!! ;) P
print(tokenizer.decode(tokens))
