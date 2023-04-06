import pandas as pd
import torch
import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer

from dataset import StackOverflowGPTDataset

def train(chatData, model, optim, device):

    epochs = 12

    for i in range(epochs):
        torch.cuda.empty_cache()
        l = 0
        for batch, data in tqdm.tqdm(enumerate(chatData), total=len(chatData)):
            X, a = data["input_ids"], data["attention_mask"]
            X = X.to(device)
            a = a.to(device)
            optim.zero_grad()
            loss = model(X, attention_mask=a, labels=X).loss
            loss.backward()
            optim.step()
            l += loss.item()
        print(l/len(chatData))
        torch.save(model.state_dict(), "model_state.pt")
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({
    "eos_token": "<|endoftext|>",
    "bos_token": "<|startoftext|>",
    "sep_token": "<|sep|>",
    "pad_token": "<|pad|>"
})

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))
model = model.to(device)
def collate_fn(batch):
    # Get the input_ids and attention_masks from the batch
    input_ids = [item['input_ids'].tolist() for item in batch]
    attention_masks = [item['attention_mask'].tolist() for item in batch]

    # Pad the input_ids and attention_masks to the same length
    max_length = 512
    padded_input_ids = [seq + [0] * (max_length - len(seq)) for seq in input_ids]
    padded_attention_masks = [seq + [0] * (max_length - len(seq)) for seq in attention_masks]

    # Convert the padded input_ids and attention_masks to tensors
    input_ids = torch.tensor(padded_input_ids)
    attention_masks = torch.tensor(padded_attention_masks)

    # Create a dictionary containing the input_ids, attention_masks, and labels
    batch_dict = {'input_ids': input_ids, 'attention_mask': attention_masks}

    return batch_dict
questions_df = pd.read_csv('data/Questions_cleaned.csv', encoding='latin-1')
answers_df = pd.read_csv('data/Answers_cleaned.csv', encoding='latin-1')
dataset = StackOverflowGPTDataset(questions_df, answers_df, tokenizer)
loader =  DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

model.train()

optim = Adam(model.parameters(), lr=1e-3)

print("training .... ")
train(loader, model, optim, device)
