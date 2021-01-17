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
length = 22
repetition_penalty = 1.5
temperature = 1


def reply(update, context):
    tokens = tokenizer.encode(update.message.text)
    for _ in range(length):
        logits = model(torch.tensor([tokens], device=gpu))[0]
        next_token_logits = logits[0, -1, :] / temperature
        for token in set(tokens):
            next_token_logits[token] /= repetition_penalty
        tokens += [torch.argmax(next_token_logits).item()]
    context.bot.send_message(chat_id=update.effective_chat.id, text=tokenizer.decode(tokens))


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
updater = Updater(TOKEN, request_kwargs=REQUEST_KWARGS, use_context=True)
dispatcher = updater.dispatcher
echo_handler = MessageHandler(Filters.text & (~Filters.command), reply)
dispatcher.add_handler(echo_handler)
updater.start_polling()
