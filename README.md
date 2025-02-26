# 🚀 GPT-2 Fine-Tuning Locally

This project fine-tunes a **GPT-2 language model** locally on a **custom text dataset** using Hugging Face's `transformers` library.  
The implementation follows a **modular class-based approach** for better scalability and maintainability.

## **📌 Features**
✅ Fine-tunes GPT-2 on a custom dataset  
✅ Uses Hugging Face's `Trainer` API for easy training  
✅ Saves the trained model locally for inference  

---

## **📂 Project Structure**
```
GPT2-FineTuning/
│── training_data.txt         # Your custom dataset (plain text file)
│── dataset_loader.py         # Used to load data and tokenization
│── model_trainer.py          # Used for training the model
│── fine_tune.py              # Used to trigger fine-tuning
│── example.py                # Example to use fine-tuned model
│── gpt2-finetuned/           # Folder where the fine-tuned model is saved
│── README.md                 # Project documentation
```

---

## **📦 Installation**
### **Step 1: Set Up Python Environment**
Make sure you have **Python 3.8+** installed, then create a virtual environment:
```bash
python -m venv fine-tune-env
source fine-tune-env/bin/activate  # Mac/Linux
```

### **Step 2: Install Dependencies**
```bash
pip install torch transformers datasets
```

---

## **📌 How to Train the Model**
1️⃣ Place your **text dataset** in a file named `data.txt`.  
2️⃣ Run the fine tuning script:
```bash
python fine_tune.py
```
3️⃣ The **fine-tuned model** will be saved in `./gpt2-finetuned/`.  

---

## **⚙️ How It Works**
The training process is structured into three main classes:

| Class | Function |
|--------|----------|
| `DatasetLoader` | Loads & tokenizes the dataset |
| `ModelTrainer` | Initializes GPT-2 and fine-tunes it |
| `FineTuneGPT2` | Manages the full training pipeline |

📌 **Training settings** (batch size, epochs, model name, etc.) can be modified in `training_script.py`.

---

## **🛠 Example Usage (Inference)**
After training, you can load and test your model:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_path = "./gpt2-finetuned"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

prompt = "The sky is"
inputs = tokenizer(prompt, return_tensors="pt")

output = model.generate(**inputs, max_length=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

🚀 **Happy Fine-Tuning!**

