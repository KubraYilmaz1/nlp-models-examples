import logging

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from telegram.ext import Updater, MessageHandler, Filters
gpu = torch.device('cuda')
tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")
model = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2").to(device=gpu)

TOKEN = 'TOKEN'
REQUEST_KWARGS = {
    'proxy_url': 'socks5://IP:PORT',
    'urllib3_proxy_kwargs': {
        'username': 'LOGIN',
        'password': 'PASSWORD',
    }
}
length = 100
repetition_penalty = 1.5
temperature = 1
threshold = 0.1


def reply(update, context):
    tokens = tokenizer.encode(update.message.text)
    for _ in range(length):
        with torch.no_grad():
            logits = model(torch.tensor([tokens], device=gpu))[0]
            next_token_logits = logits[0, -1, :] / temperature
            for token in set(tokens):
                next_token_logits[token] /= repetition_penalty
            sorted_probs, sorted_indices = torch.sort(torch.softmax(next_token_logits, dim=-1), descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            new_idx = torch.sum(cumulative_probs < 0.3) + 1
            sorted_probs, sorted_indices = sorted_probs[:new_idx], sorted_indices[:new_idx]
            sorted_probs /= torch.sum(sorted_probs)
            tokens += [sorted_indices[torch.multinomial(sorted_probs, num_samples=1).item()]]
    context.bot.send_message(chat_id=update.effective_chat.id, text=tokenizer.decode(tokens))


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
updater = Updater(TOKEN, request_kwargs=REQUEST_KWARGS, use_context=True)
dispatcher = updater.dispatcher
echo_handler = MessageHandler(Filters.text & (~Filters.command), reply)
dispatcher.add_handler(echo_handler)
updater.start_polling()
