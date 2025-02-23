
from dataset_loader import DatasetLoader
from model_trainer import ModelTrainer

class FineTuneGPT2:
    """Manages the full fine-tuning pipeline."""
    
    def __init__(self, dataset_path, output_dir="./gpt2-finetuned"):
        self.dataset_path = dataset_path
        self.output_dir = output_dir

    def run(self):
        """Runs the full fine-tuning pipeline."""
        # Step 1: Load and preprocess data
        data_loader = DatasetLoader(self.dataset_path)
        tokenized_dataset = data_loader.preprocess_data()

        # Step 2: Train the model
        trainer = ModelTrainer(output_dir=self.output_dir)
        trainer.train(tokenized_dataset)


if __name__ == "__main__":
    # Provide your dataset file (plain text file)
    dataset_path = "training_data.txt"

    # Run fine-tuning
    fine_tuner = FineTuneGPT2(dataset_path)
    fine_tuner.run()