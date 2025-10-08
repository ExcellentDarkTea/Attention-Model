import os
import sys
import glob
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.model_selection import KFold
from sklearn.metrics import (
    f1_score, accuracy_score, recall_score, precision_score
)

from scr import utils, models_classes, preprocess


def main():
    path = 'DATA'
    files = glob.glob(os.path.join(path, '*.csv'))

    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    model_types = [
        models_classes.SimpleLSTM,
        models_classes.StressClassifierWithFusion,
        models_classes.TCN_net
    ]

    scores_per_model = {"model_type": [], "scores": []}

    for model_class in model_types:
        print(f"\n=== Evaluating {model_class.__name__} ===")

        scores_average = {
            "accuracy_test": [], "f1_test": [], "recall_test": [], "precision_test": [],
            "accuracy_train": [], "f1_train": [], "recall_train": [], "precision_train": []
        }

        for fold, (train_idx, test_idx) in enumerate(kf.split(files)):
            print(f"\n===== Fold {fold + 1} =====")

            file_train = [files[i] for i in train_idx]
            file_test = [files[i] for i in test_idx]

            X_train, y_train = torch.tensor([]), torch.tensor([])
            X_test, y_test = torch.tensor([]), torch.tensor([])
            scalers = []

            for file in files:
                df = pd.read_csv(file)
                df = df.dropna(subset=['HR', 'EDA', 'TEMP', 'ACC_X', 'ACC_Y', 'ACC_Z'])
                df['time'] = pd.to_datetime(df['time'])
                df = preprocess.trim_after_last_one(df, label_col='stress')

                df_standardized, scaler_new = preprocess.scaler_per_user(df)
                scalers.append(scaler_new)

                X, y, _ = preprocess.create_dataset(
                    df_standardized, window_size=60, overlap_step=25, load_threshold=10
                )

                if file in file_train:
                    X_train = torch.cat((X_train, X), dim=0) if X_train.numel() else X
                    y_train = torch.cat((y_train, y), dim=0) if y_train.numel() else y
                else:
                    X_test = torch.cat((X_test, X), dim=0) if X_test.numel() else X
                    y_test = torch.cat((y_test, y), dim=0) if y_test.numel() else y

            # Dataloaders
            BATCH_SIZE = 32
            train_dataset = torch.utils.data.TensorDataset(X_train, y_train.long())
            test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            # Model setup
            model = model_class()
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            EPOCHS = 5

            save_path = f"best_model_{model_class.__name__}_fold{fold+1}.pth"
            scores = utils.train_model(model, train_loader, test_loader, criterion, optimizer,
                                       num_epochs=EPOCHS, save_path=save_path)

            # Evaluate best model
            model_best = model_class()
            model_best.load_state_dict(torch.load(save_path))
            model_best.eval()

            y_preds_test, y_trues_test = [], []
            y_preds_train, y_trues_train = [], []

            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    y_pred = model_best(X_batch)
                    y_pred = (y_pred > 0.5).float()
                    y_preds_test.extend(y_pred.cpu().numpy())
                    y_trues_test.extend(y_batch.cpu().numpy())

                for X_batch, y_batch in train_loader:
                    y_pred = model_best(X_batch)
                    y_pred = (y_pred > 0.5).float()
                    y_pred_np = y_pred.cpu().numpy()
                    if y_pred_np.ndim == 0:
                        y_pred_np = y_pred_np.reshape(1)
                    y_preds_train.extend(y_pred_np)
                    y_trues_train.extend(y_batch.cpu().numpy())

            # Metrics
            scores_average['accuracy_test'].append(accuracy_score(y_trues_test, y_preds_test))
            scores_average['f1_test'].append(f1_score(y_trues_test, y_preds_test))
            scores_average['recall_test'].append(recall_score(y_trues_test, y_preds_test))
            scores_average['precision_test'].append(precision_score(y_trues_test, y_preds_test))

            scores_average['accuracy_train'].append(accuracy_score(y_trues_train, y_preds_train))
            scores_average['f1_train'].append(f1_score(y_trues_train, y_preds_train))
            scores_average['recall_train'].append(recall_score(y_trues_train, y_preds_train))
            scores_average['precision_train'].append(precision_score(y_trues_train, y_preds_train))

        scores_per_model["model_type"].append(model_class.__name__)
        scores_per_model["scores"].append(scores_average)

    # Final summary
    print("\n=== Average scores over K=5 folds ===")
    for model_name, scores in zip(scores_per_model["model_type"], scores_per_model["scores"]):
        print(f"\nModel: {model_name}")
        for metric, value in scores.items():
            print(f"  {metric}: {np.mean(value):.4f} ± {np.std(value):.4f}")

    # # Save results
    # with open("output_kFold.json", "w") as json_file:
    #     json.dump(scores_per_model, json_file, indent=4)


if __name__ == "__main__":
    main()
