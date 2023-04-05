import torch
from torch.utils.data import Dataset

class StackOverflowGPTDataset(Dataset):
    """
    Dataset for StackOverflow GPT model.
    """
    def __init__(self, questions_df, answers_df, tokenizer):
        self.questions_df = questions_df
        self.answers_df = answers_df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.answers_df)

    def __getitem__(self, idx):
        answer_body = self.answers_df.iloc[idx]['Body']
        q_id = self.answers_df.iloc[idx]['ParentId']
        question_body = self.questions_df.loc[self.questions_df['Id'] == q_id]['Body'].values[0]
        input_text =  "<|startoftext|>"+ question_body + "<|sep|>" + answer_body + "<|endoftext|>"
        encoded = self.tokenizer(input_text, truncation=True, max_length=512)
        return {
            "input_ids": torch.tensor(encoded['input_ids']),
            "attention_mask": torch.tensor(encoded['attention_mask']),
        }

if __name__ == "__main__":
    from transformers import GPT2Tokenizer
    import pandas as pd
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({
        "eos_token": "<|endoftext|>",
        "bos_token": "<|startoftext|>",
        "sep_token": "<|sep|>",
        "pad_token": "<|pad|>"
    })
    questions_df = pd.read_csv('data/Questions_cleaned.csv', encoding='latin-1')
    answers_df = pd.read_csv('data/Answers_cleaned.csv', encoding='latin-1')
    dataset = StackOverflowGPTDataset(questions_df, answers_df, tokenizer)
    print(dataset[0])
    decoded = tokenizer.decode(dataset[0]['input_ids'])
    print("Decoded string: ", decoded)
