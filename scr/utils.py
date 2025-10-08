import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix



def evaluate_model(model, data_loader, device):
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch).view(-1)
            preds = (outputs > 0.5).int()
            all_preds.append(preds.cpu())
            all_labels.append(y_batch.cpu())
    return torch.cat(all_labels), torch.cat(all_preds)


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, save_path='best_model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_f1 = 0.0
    best_model = None

    metrics = {
        'train_loss': [], 'test_loss': [],
        'train_f1': [], 'test_f1': [],
        'train_acc': [], 'test_acc': [],
        'train_precision': [], 'test_precision': [],
        'train_recall': [], 'test_recall': []
    }

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch).view(-1)
            loss = criterion(outputs, y_batch.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        metrics['train_loss'].append(total_loss / len(train_loader))

        # Evaluation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).view(-1)
                loss = criterion(outputs, y_batch.float())
                test_loss += loss.item()
        metrics['test_loss'].append(test_loss / len(test_loader))

        # Metrics
        y_true_train, y_pred_train = evaluate_model(model, train_loader, device)
        y_true_test, y_pred_test = evaluate_model(model, test_loader, device)

        metrics['train_f1'].append(f1_score(y_true_train, y_pred_train))
        metrics['test_f1'].append(f1_score(y_true_test, y_pred_test))
        metrics['train_acc'].append(accuracy_score(y_true_train, y_pred_train))
        metrics['test_acc'].append(accuracy_score(y_true_test, y_pred_test))
        metrics['train_precision'].append(precision_score(y_true_train, y_pred_train))
        metrics['test_precision'].append(precision_score(y_true_test, y_pred_test))
        metrics['train_recall'].append(recall_score(y_true_train, y_pred_train))
        metrics['test_recall'].append(recall_score(y_true_test, y_pred_test))

        test_f1 = metrics['test_f1'][-1]

        if test_f1 > best_f1:
                best_f1 = test_f1
                torch.save(model.state_dict(), save_path)
                print(f"!!! Best model updated at epoch {epoch+1} with Test F1: {best_f1:.4f} !!!")

        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {metrics['train_loss'][-1]:.4f} - Test Loss: {metrics['test_loss'][-1]:.4f} - "
              f"Train F1: {metrics['train_f1'][-1]:.4f} - Test F1: {metrics['test_f1'][-1]:.4f}")
        print(f"Confusion matrix (Test):\n{confusion_matrix(y_true_test, y_pred_test)}")

    #save the best model
    torch.save(best_model, 'best_model.pth')

    return metrics, best_model
