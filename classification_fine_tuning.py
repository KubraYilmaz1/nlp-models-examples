import glob
import os
import random
import re
import shutil
import sys
import tarfile
from io import BytesIO
import requests
import torch
from colorama import Fore
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW

# ============================================= PREPARING DATASET ======================================================
base_dir = 'data/aclImdb'
if not os.path.isdir('data'):
    r = requests.get('https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
    tar = tarfile.open(fileobj=BytesIO(r.content))
    tar.extractall(path="data")
    tar.close()
    train_pos_files = glob.glob(base_dir + '/train/pos/*.txt')
    train_neg_files = glob.glob(base_dir + '/train/neg/*.txt')
    random.shuffle(train_pos_files)
    random.shuffle(train_neg_files)
    val_pos_files = train_pos_files[:1000]
    val_neg_files = train_neg_files[:1000]
    os.makedirs(base_dir + '/val/pos', exist_ok=True)
    os.makedirs(base_dir + '/val/neg', exist_ok=True)
    for f in val_pos_files:
        shutil.move(f, base_dir + '/val/pos')
    for f in val_neg_files:
        shutil.move(f, base_dir + '/val/neg')

tokenizer = T5Tokenizer.from_pretrained('t5-base')
gpu = torch.device('cuda')


class ImdbDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=512):
        self.pos_file_path = os.path.join(data_dir, type_path, 'pos')
        self.neg_file_path = os.path.join(data_dir, type_path, 'neg')
        self.pos_files = glob.glob("%s/*.txt" % self.pos_file_path)
        self.neg_files = glob.glob("%s/*.txt" % self.neg_file_path)
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self.buil_examples_from_files(self.pos_files, 'positive')
        self.buil_examples_from_files(self.neg_files, 'negative')

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()
        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def buil_examples_from_files(self, files, sentiment):
        replace_no_space = re.compile("[.;:!\'?,\"()\[\]]")
        replace_with_space = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

        for path in files:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            line = text.strip()
            line = replace_no_space.sub("", line)
            line = replace_with_space.sub("", line)
            line = line + ' </s>'
            target = sentiment + " </s>"
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [line], max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length=2, padding='max_length', truncation=True, return_tensors="pt")
            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)


dataloader_train = DataLoader(ImdbDataset(tokenizer, base_dir, 'train', max_len=512), batch_size=12, drop_last=True, shuffle=True)
dataloader_val = DataLoader(ImdbDataset(tokenizer, base_dir, 'val', max_len=512), batch_size=12)
# ============================================ HYPERPARAMETERS =========================================================
learning_rate = 3e-4
adam_epsilon = 1e-8
epochs = 4
# ================================================= MODEL ==============================================================
model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device=gpu)
no_decay = ["bias", "LayerNorm.weight"]
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon)
# ================================================ TRAINING MODEL ======================================================
for epoch in range(1, epochs + 1):
    # ============================================ TRAINING ============================================================
    print("Training epoch ", str(epoch))
    model.train()
    tr_loss = 0
    nb_tr_steps = 0
    for batch in tqdm(dataloader_train,
                      position=0, leave=True,
                      file=sys.stdout, bar_format="{l_bar}%s{bar:50}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100
        optimizer.zero_grad()
        outputs = model(batch["source_ids"].to(device=gpu),
                        attention_mask=batch["source_mask"].to(device=gpu),
                        decoder_attention_mask=batch['target_mask'].to(device=gpu),
                        labels=lm_labels.to(device=gpu), return_dict=False)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()
        nb_tr_steps += 1
    print(f"\nTraining loss={tr_loss / nb_tr_steps:.4f}")
    # ============================================ VALIDATION ==========================================================
    model.eval()
    val_loss = 0
    nb_val_steps = 0
    val_outputs = []
    val_targets = []
    for batch in tqdm(dataloader_val,
                      position=0, leave=True,
                      file=sys.stdout, bar_format="{l_bar}%s{bar:50}%s{r_bar}" % (Fore.BLUE, Fore.RESET)):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100
        outputs = model(batch["source_ids"].to(device=gpu),
                        attention_mask=batch["source_mask"].to(device=gpu),
                        decoder_attention_mask=batch['target_mask'].to(device=gpu),
                        labels=lm_labels.to(device=gpu), return_dict=False)
        val_loss += outputs[0].item()
        nb_val_steps += 1
        outs = model.generate(input_ids=batch['source_ids'].cuda(),
                              attention_mask=batch['source_mask'].cuda(),
                              max_length=2)
        dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]
        val_outputs.extend(dec)
        val_targets.extend(target)
    print(f"\nValidation loss={val_loss / nb_val_steps:.4f}\nValidation accuracy={metrics.accuracy_score(val_targets, val_outputs)}")
torch.save(model.state_dict(), "./weights.pth")
# ============================================ TESTING =================================================================
model.load_state_dict(torch.load("./weights.pth"))
dataset_test = ImdbDataset(tokenizer, base_dir, 'test', max_len=512)
loader = DataLoader(dataset_test, batch_size=12)
model.eval()
outputs = []
targets = []
for batch in tqdm(loader, position=0, leave=True,
                  file=sys.stdout, bar_format="{l_bar}%s{bar:50}%s{r_bar}" % (Fore.RED, Fore.RESET)):
    outs = model.generate(input_ids=batch['source_ids'].cuda(),
                          attention_mask=batch['source_mask'].cuda(),
                          max_length=2)
    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
    target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]
    outputs.extend(dec)
    targets.extend(target)
print("Test accuracy:", metrics.accuracy_score(targets, outputs))
print(metrics.classification_report(targets, outputs))
