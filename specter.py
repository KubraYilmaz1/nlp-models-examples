import semanticscholar as sch
import torch
from transformers import AutoTokenizer, AutoModel
import itertools
import torch.nn.functional as F
from pprint import pprint

gpu = torch.device('cuda')

# sample papers
paper_ids = ['df2b0e26d0599ce3e70df8a9da02e51594e0e992',  # ”BERT”
             '6b85b63579a916f705a8e10a49bd8d849d91b1fc',  # ”GPT-3”
             '077f8329a7b6fa3b7c877a57b81eb6c18b5f87de',  # ”RoBERTa”
             '8e787e925eeb7ad735a228b2b1e8dd6d9620be83']  # A Clinical paper

# get paper title and abstract
papers = [sch.paper(p_id, timeout=10) for p_id in paper_ids]
paper_names = ["BERT", "GPT-3", "RoBERTa", "Clinical Course"]
paper_texts = [p['title'] + ' ' + p['abstract'] for p in papers]

# load model and tokenizer
model = AutoModel.from_pretrained("allenai/specter").to(device=gpu)
tokenizer = AutoTokenizer.from_pretrained("allenai/specter")

# preprocess raw text
inputs = [tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device=gpu) for text in paper_texts]

# get embeddings
embeddings = [model(**input).last_hidden_state[0, 0, :].detach() for input in inputs]

# get pairwise similarities between papers
pairs = list(itertools.combinations(enumerate(embeddings), 2))

similarities = {}
for pair in pairs:
    key = f"{paper_names[pair[0][0]]} # {paper_names[pair[1][0]]}"
    similarity_score = F.cosine_similarity(pair[0][1].unsqueeze(0), pair[1][1].unsqueeze(0)).item()
    similarities[key] = f'{similarity_score:.3f}'

# print results
#
# {'BERT # Clinical Course': '0.453',
#  'BERT # GPT-3': '0.853',
#  'BERT # RoBERTa': '0.792',
#  'GPT-3 # Clinical Course': '0.463',
#  'GPT-3 # RoBERTa': '0.823',
#  'RoBERTa # Clinical Course': '0.441'}
pprint(similarities, width=1)
