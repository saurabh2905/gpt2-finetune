from transformers import (
    GPT2LMHeadModel, 
    TrainingArguments, 
    Trainer
)

class ModelTrainer:
    """Handles GPT-2 fine-tuning using Hugging Face Trainer API."""
    
    def __init__(self, model_name="gpt2", output_dir="./gpt2-finetuned", num_train_epochs=3, batch_size=2):
        self.model_name = model_name
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size

        # Load model
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

        # Set up training arguments
        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="no",  # No evaluation dataset provided
            save_strategy="epoch",
            per_device_train_batch_size=self.batch_size,
            num_train_epochs=self.num_train_epochs,
            logging_dir="./logs",
            logging_steps=10,
            save_total_limit=2
        )

    def train(self, train_dataset):
        """Trains the GPT-2 model using the Trainer API."""
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset
        )

        print("🚀 Training started...")
        trainer.train()
        print("✅ Training completed!")

        # Save model
        self.model.save_pretrained(self.output_dir)
        print(f"📁 Model saved at {self.output_dir}")
