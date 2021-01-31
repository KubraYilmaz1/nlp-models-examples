import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

test_data = [{
    'sentence1': 'The rain in Spain falls mainly on the plain.',
    'sentence2': 'It mostly rains on the flat lands of Spain.'
}, {
    'sentence1': 'Look I fine tuned BERT.',
    'sentence2': 'Is it working? This does not match.'
}]

gpu = torch.device('cuda')
model = T5ForConditionalGeneration.from_pretrained('t5-large').to(gpu)
tokenizer = T5Tokenizer.from_pretrained('t5-large')
for pair in test_data:
    tokens_input = tokenizer.encode(text="mrpc sentence1: " + pair['sentence1'] + " sentence2: " + pair['sentence2'],
                                    return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(tokens_input.to(gpu), min_length=1, max_length=10, length_penalty=4.0)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    # equivalent
    # not_equivalent
    print(summary)
    print()
