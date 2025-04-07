import os
# Suppress Hugging Face symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from torch.optim import AdamW  # Correct import for AdamW
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

tokenizedreviewfolders = ["./tokenizedsamplepos", "./tokenizedsampleneg"]

data = [
        "I liked the movie",
        "I hated the movie",
        "The movie was okay",
        "The movie was fantastic",
    ]
labels = [1, 0, 1, 1]  # 1 for positive, 0 for negative
df = pd.DataFrame({'review': data, 'label': labels})  # Add column names
print(df)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoded_data = tokenizer(
    df['review'].tolist(),                # Use the 'review' column
    padding=True,                         # Padding automatique
    truncation=True,                      # Tronque si > max_length
    max_length=64,                        # Longueur max des séquences
    return_tensors='pt'                   # Retourne des tenseurs PyTorch
)
labels = torch.tensor(df['label'].tolist())  # Use the 'label' column
print(labels)
dataset = TensorDataset(
    encoded_data['input_ids'],
    encoded_data['attention_mask'],
    labels
)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
print("DataLoader created with batch size 8")

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

optimizer = AdamW(model.parameters(), lr=2e-5)

num_epochs = 4
num_training_steps = num_epochs * len(dataloader)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

model.train()

for epoch in range(num_epochs):
    loop = tqdm(dataloader, leave=True)
    total_loss = 0

    for batch in loop:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        logits = outputs.logits

        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # Mise à jour de la barre de progression
        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} finished, avg loss = {total_loss / len(dataloader):.4f}")

#output_folder_pos = "./tensorized_data/pos"
#output_folder_neg = "./tensorized_data/neg"

#tensorized_samples_pos = tensorize_samples([tokenizedreviewfolders[0]], output_folder_pos)
#tensorized_samples_neg = tensorize_samples([tokenizedreviewfolders[1]], output_folder_neg)