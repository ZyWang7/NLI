import torch
from transformers import RobertaForSequenceClassification
from transformers import AdamW
from torch.utils.data import DataLoader
from nli_dataset import NliDataset
from tqdm import tqdm
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def main():
    # get model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base').to(device)

    # get dataset
    train_dataset = NliDataset(csv_file="train_data.csv", max_length=256)
    val_dataset = NliDataset(csv_file="test_data.csv", max_length=256)

    train_loader = DataLoader(
                        dataset=train_dataset,
                        batch_size=128,
                        num_workers=4,
                        prefetch_factor=2,
                        shuffle=True,
                        drop_last=False
                    )

    val_loader = DataLoader(
                        dataset=val_dataset,
                        batch_size=128,
                        num_workers=4,
                        prefetch_factor=2,
                        drop_last=False
                    )

    # Define optimizer and loss function
    optimizer = AdamW(model.parameters(),
                    betas=(0.9, 0.98),  # according to RoBERTa paper
                    lr=1e-4, 
                    weight_decay=5e-2)

    linear_sl = lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=1)
    cos_sl = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    scheduler = lr_scheduler.SequentialLR(optimizer, schedulers=[linear_sl, cos_sl], milestones=[2])

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)


    # Training loop
    total_best_val_acc = 0
    for epoch in range(20):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = loss_fn(outputs.logits, labels)
            train_loss += loss.item()
            # loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
        
        scheduler.step()
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader)}')
        
        # Validation loop
        model.eval()
        val_loss = 0
        # Initialize lists to store true labels and predicted labels
        true_labels_list = []
        preds_list = []
        for batch in val_loader:
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

        if val_acc > total_best_val_acc:
            total_best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        print(f'Validation: Epoch {epoch + 1}, Loss: {val_loss / len(val_loader)}, Acc: {val_acc}, F1: {f1}, Precision: {precision}, Recall: {recall}')


if __name__ == '__main__':
    main()