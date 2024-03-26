import torch
from transformers import RobertaForSequenceClassification

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from nli_dataset import NliDataset

def evaluate(model_weight):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    model.load_state_dict(torch.load('best_model.pth'))
    model.to(device)

    eval_dataset = NliDataset(csv_file="Data/dev.csv", max_length=256)
    eval_loader = DataLoader(
                        dataset=eval_dataset,
                        batch_size=64,
                        num_workers=4,
                        prefetch_factor=2,
                        shuffle=True,
                        drop_last=False
                    )

    loss_fn = torch.nn.CrossEntropyLoss()

    # Initialize lists to store true labels and predicted labels
    true_labels_list = []
    preds_list = []
    val_loss = 0
    # Validation loop
    model.eval()
    for batch in tqdm(eval_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = loss_fn(outputs.logits, labels)
            val_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)

            # Append true labels and predicted labels to the lists
            true_labels_list.extend(labels.cpu().numpy())
            preds_list.extend(preds.cpu().numpy())

    # Calculate accuracy and F1-score
    # Calculate evaluation metrics
    val_acc = accuracy_score(true_labels_list, preds_list)
    f1 = f1_score(true_labels_list, preds_list, average='weighted')
    precision = precision_score(true_labels_list, preds_list, average='weighted')
    recall = recall_score(true_labels_list, preds_list, average='weighted')

    print("Validation Loss:", val_loss.item())
    print(f'Validation: Loss: {val_loss / len(eval_loader)}, Acc: {val_acc}, F1: {f1}, Precision: {precision}, Recall: {recall}')

if __name__ == '__main__':
    evaluate('best_model.pth')

