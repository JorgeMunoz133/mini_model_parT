"""
evaluate.py
===========
Checks how well the trained model actually does on data it has never
seen: overall accuracy, a confusion matrix, ROC curves per class, and a
"fingerprint" similarity check. See lessons/06_evaluating_the_model.md
for the full explanation.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

CLASS_NAMES = ["Hbb", "Hcc", "QCD"]


def evaluate_model(model, test_loader, device):
    model.eval()

    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            outputs = model(batch_x)
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    test_acc = 100. * (all_preds == all_labels).mean()
    print(f"Final Test Accuracy: {test_acc:.2f}%")

    # --- Confusion matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f"{c} ({i})" for i, c in enumerate(CLASS_NAMES)],
                yticklabels=[f"{c} ({i})" for i, c in enumerate(CLASS_NAMES)])
    plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
    plt.ylabel('True Class', fontsize=12, fontweight='bold')
    plt.title('MiniParT Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()
    print("Saved confusion_matrix.png")

    # --- ROC curves, one-vs-rest ---
    y_test_bin = label_binarize(all_labels, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]

    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green']
    for i, color, name in zip(range(n_classes), colors, CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} vs Rest (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guessing')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Background Efficiency)', fontsize=12)
    plt.ylabel('True Positive Rate (Signal Efficiency)', fontsize=12)
    plt.title('MiniParT ROC Curves', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curves.png")
    plt.close()
    print("Saved roc_curves.png")

    return all_labels, all_preds, all_probs


def fingerprint_similarity(model, X_test_scaled, y_test, device):
    """
    Grabs one Hbb, one Hcc, and one QCD event from the test set and
    compares the model's internal 64-number "fingerprint" of each,
    using cosine similarity. See the "Bonus" section of
    lessons/06_evaluating_the_model.md.
    """
    model.eval()

    events = {0: None, 1: None, 2: None}
    for i in range(len(y_test)):
        label = int(y_test[i])
        if events[label] is None:
            events[label] = torch.tensor(X_test_scaled[i:i + 1], dtype=torch.float32).to(device)
        if all(v is not None for v in events.values()):
            break

    with torch.no_grad():
        fps = {label: model.get_fingerprint(t) for label, t in events.items()}

    def sim(a, b):
        return F.cosine_similarity(fps[a], fps[b]).item()

    data = {
        "Hbb": [1.0, sim(0, 1), sim(0, 2)],
        "Hcc": [sim(0, 1), 1.0, sim(1, 2)],
        "QCD": [sim(0, 2), sim(1, 2), 1.0],
    }
    df_sim = pd.DataFrame(data, index=["Hbb", "Hcc", "QCD"])
    print("\nCosine Similarity Matrix (1 = Identical, -1 = Opposite):\n")
    print(df_sim.round(3))
    return df_sim
