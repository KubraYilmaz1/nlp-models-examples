from transformers import BertTokenizer, BertModel
import pandas as pd
import torch
from scipy.spatial.distance import cosine

model = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def prepare_word(text):
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("[CLS] " + text + " [SEP]"))
    segments_ids = [1] * len(indexed_tokens)
    return torch.tensor([indexed_tokens]), torch.tensor([segments_ids])


@torch.no_grad()
def get_embeddings(tokens_tensor, segments_tensors):
    outputs = model(tokens_tensor, segments_tensors)
    # take the first index from the output - last_hidden_state
    # take the first element from the batch
    # skip [CLS] token, so just take the second embedding
    return outputs[0][0][1].numpy()


words = ["queen", "king", "crown", "girl", "woman", "man", "river"]
embeddings = [get_embeddings(*prepare_word(word)) for word in words]

distances = [[words[0], word2, 1 - cosine(embeddings[0], embed2)] for word2, embed2 in zip(words, embeddings)]

distances_df = pd.DataFrame(distances, columns=['word1', 'word2', 'distance'])

#          word1  word2  distance
#       0  queen  queen  1.000000
#       1  queen   king  0.864386
#       2  queen  crown  0.878993
#       3  queen   girl  0.744251
#       4  queen  woman  0.780992
#       5  queen    man  0.775225
#       6  queen  river  0.489919
print(distances_df)
