import logging

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from telegram.ext import Updater, MessageHandler, Filters, CommandHandler
from collections import Counter

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
length = 200
repetition_penalty = 0.1
temperature = 1
threshold = 0.3
n = 3

user_threshold = {}
user_temperature = {}
user_repetition_penalty = {}
user_n_gram = {}


def reply(update, context):
    print('Threshold', user_threshold.get(update.effective_chat.id, threshold))
    print('Temperature', user_temperature.get(update.effective_chat.id, temperature))
    print('Repetition penalty', user_repetition_penalty.get(update.effective_chat.id, repetition_penalty))
    tokens = tokenizer.encode(update.message.text)
    with torch.no_grad():
        for _ in range(length):
            logits = model(torch.tensor([tokens], device=gpu))[0]
            effective_n = user_n_gram.get(update.effective_chat.id, n)
            ngrams = zip(*[tokens[i:] for i in range(effective_n)])
            last_ngram_count = Counter(ngrams).get(tuple(tokens[-effective_n:]))
            softmax_temp = 1
            if last_ngram_count is not None and last_ngram_count > 1:
                softmax_temp = (1 + user_repetition_penalty.get(update.effective_chat.id, repetition_penalty)) ** (last_ngram_count - 1)
            effective_temp = user_temperature.get(update.effective_chat.id, temperature) * softmax_temp
            next_token_logits = logits[0, -1, :] / effective_temp
            sorted_probs, sorted_indices = torch.sort(torch.softmax(next_token_logits, dim=-1), descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            new_idx = torch.sum(cumulative_probs < user_threshold.get(update.effective_chat.id, threshold)) + 1
            sorted_probs, sorted_indices = sorted_probs[:new_idx], sorted_indices[:new_idx]
            sorted_probs /= torch.sum(sorted_probs)
            res = sorted_indices[torch.multinomial(sorted_probs, num_samples=1).item()]
            tokens += [res.item()]
    context.bot.send_message(chat_id=update.effective_chat.id, text=tokenizer.decode(tokens))


def set_threshold(update, context):
    user_threshold[update.effective_chat.id] = float(context.args[0])
    context.bot.send_message(chat_id=update.effective_chat.id, text="Done!")


def set_repetition_penalty(update, context):
    user_repetition_penalty[update.effective_chat.id] = float(context.args[0])
    context.bot.send_message(chat_id=update.effective_chat.id, text="Done!")


def set_temperature(update, context):
    user_temperature[update.effective_chat.id] = float(context.args[0])
    context.bot.send_message(chat_id=update.effective_chat.id, text="Done!")


def set_gram(update, context):
    user_n_gram[update.effective_chat.id] = float(context.args[0])
    context.bot.send_message(chat_id=update.effective_chat.id, text="Done!")


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
updater = Updater(TOKEN, request_kwargs=REQUEST_KWARGS, use_context=True)
dispatcher = updater.dispatcher
dispatcher.add_handler(CommandHandler('th', set_threshold))
dispatcher.add_handler(CommandHandler('temp', set_temperature))
dispatcher.add_handler(CommandHandler('rp', set_repetition_penalty))
dispatcher.add_handler(CommandHandler('gram', set_gram))
dispatcher.add_handler(MessageHandler(Filters.text & (~Filters.command), reply))
updater.start_polling()
