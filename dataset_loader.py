from transformers import GPT2Tokenizer
from datasets import load_dataset

class DatasetLoader:
    """Loads and tokenizes the dataset for training GPT-2."""
    
    def __init__(self, dataset_path, tokenizer_name="gpt2", max_length=512):
        self.dataset_path = dataset_path
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # Ensure padding token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_data(self):
        """Loads text data from a file and returns a dataset."""
        dataset = load_dataset("text", data_files={"train": self.dataset_path})
        return dataset

    def tokenize_function(self, examples):
        """Tokenizes the text and adds labels for training."""
        tokenized_output = self.tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_length
        )
        tokenized_output["labels"] = tokenized_output["input_ids"].copy()  # Labels are input_ids
        return tokenized_output

    def preprocess_data(self):
        """Loads and tokenizes the dataset."""
        dataset = self.load_data()
        tokenized_datasets = dataset["train"].map(self.tokenize_function, batched=True)
        return tokenized_datasets
