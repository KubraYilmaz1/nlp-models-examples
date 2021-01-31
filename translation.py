import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

context = "This image section from an infrared recording by the Spitzer telescope " \
          "shows a \"family portrait\" of countless generations of stars: the oldest stars " \
          "are seen as blue dots, while more difficult to identify are the pink-coloured " \
          "\"new-borns\" in the star delivery room."

gpu = torch.device('cuda')
model = T5ForConditionalGeneration.from_pretrained('t5-large').to(gpu)
tokenizer = T5Tokenizer.from_pretrained('t5-large')
tokens_input = tokenizer.encode(text="translate English to French: " + context, return_tensors="pt",
                                max_length=1024, truncation=True)
summary_ids = model.generate(tokens_input.to(gpu), min_length=60, max_length=1024, length_penalty=4.0)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Cette section d'image d'un enregistrement infrarouge du télescope Spitzer montre un « portrait familial »
# d'innombrables générations d'étoiles : les étoiles les plus anciennes sont observées comme des points bleus,
# tandis que les « nouveau-nés » de couleur rose dans la salle d'accouchement des étoiles sont plus difficiles à
# identifier.
print(summary)
