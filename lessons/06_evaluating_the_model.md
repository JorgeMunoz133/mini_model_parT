# Lesson 6 - Evaluating the Model

Training accuracy (Lesson 5) can lie to you - a model can look great on
data it's already memorized and still be useless on new data. This lesson
is about actually finding out whether MiniParT learned something real,
using only the 20% of data it never saw during training. All of this is in
`code/evaluate.py`.

## Step 1 - Test accuracy

```python
model.eval()
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        _, predicted = outputs.max(1)
        correct += predicted.eq(batch_y).sum().item()

test_acc = 100. * correct / total
```

- **`model.eval()`** - the opposite of `model.train()` from Lesson 5. It
  switches off dropout, so the model always gives its single best,
  consistent answer instead of the slightly-randomized version used
  during training.
- **`torch.no_grad()`** - tells PyTorch not to bother tracking how to
  compute gradients here, since we're not training, just checking
  answers. This makes evaluation faster and uses less memory.
- **`outputs.max(1)`** - remember the model outputs 3 raw scores per
  event (Lesson 4). `.max(1)` just picks out whichever of the 3 is
  highest - that becomes the model's actual predicted class.
- **`test_acc`** - the percentage of *unseen* test examples the model
  classified correctly. This is the honest report card.

## Step 2 - The confusion matrix: not just "right or wrong," but *how* wrong

Overall accuracy hides an important detail: is the model actually
struggling to tell Hbb from Hcc specifically (the genuinely hard physics
problem from Lesson 0), or is it mostly confusing signal with background
instead? A **confusion matrix** answers that by showing, for every true
class, exactly which class the model guessed:

```python
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Hbb (0)', 'Hcc (1)', 'QCD (2)'],
            yticklabels=['Hbb (0)', 'Hcc (1)', 'QCD (2)'])
```

Read it like this: each row is "events that were *actually* this class,"
each column is "events the model *guessed* were this class." A perfect
model would have big numbers only along the diagonal (true class = guessed
class) and zeros everywhere else. If you see a lot of events landing in
the Hbb-row/Hcc-column square (or vice versa), that's the model mixing up
the two Higgs decay types specifically - which, given how physically
similar bottom and charm jets are, is exactly where you'd expect it to
struggle most.

## Step 3 - ROC curves: how good is each class at different confidence thresholds

So far we've only looked at the model's single best guess. But the model
actually outputs a *confidence* for every class (via `softmax`, which
turns the 3 raw scores into 3 probabilities that add up to 1). Depending
on the physics analysis you're doing, you might want to be stricter or
looser about how confident the model needs to be before you "trust" a
guess. A **ROC curve** (Receiver Operating Characteristic) shows that
whole tradeoff at once, for one class versus everything else:

```python
fpr, tpr, _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
roc_auc = auc(fpr, tpr)
```

- **True Positive Rate** (y-axis) - of all the real Hbb events, what
  fraction did we correctly flag as Hbb, at some confidence threshold?
  Physicists often call this "signal efficiency."
- **False Positive Rate** (x-axis) - of all the events that were *not*
  actually Hbb, what fraction did we mistakenly flag as Hbb anyway?
  Physicists call this "background efficiency" (it's the rate at which
  background sneaks through).
- As you slide the confidence threshold from strict to loose, both rates
  change together, tracing out the curve. A model that's just guessing
  randomly traces the diagonal line; a genuinely useful model bulges up
  toward the top-left corner (catch real events, reject background).
- **AUC** (Area Under the Curve) boils that whole curve down to one
  number between 0 and 1: `0.5` means no better than a coin flip, `1.0`
  means a perfect classifier. It's a standard way to compare classifiers
  at a glance.

We do this once per class ("Hbb vs. everything else," "Hcc vs. everything
else," "QCD vs. everything else") - that's called a **one-vs-rest**
approach, and it's why the labels get "binarized" first
(`label_binarize`) into three separate yes/no columns.

## Bonus - Peeking inside the model with "fingerprints"

There's one more genuinely fun thing in the notebook worth understanding.
Instead of only looking at the model's *final* answer, we can grab its
internal 64-number description of an event - right after self-attention
and pooling, but *before* the final classification head (Lesson 4, Piece
3). Think of this 64-number vector as the model's own internal
"fingerprint" of what kind of event it thinks this is.

```python
def get_fingerprint(event_tensor):
    emb = model.embedding(event_tensor)
    contextualized = model.transformer(emb)
    fingerprint = contextualized.mean(dim=1)
    return fingerprint
```

If we grab one real Hbb event, one Hcc event, and one QCD event, and
compare their fingerprints using **cosine similarity** (a measure of how
similarly two vectors "point," from -1 = opposite to 1 = identical), we
get a direct, human-readable check of whether the model's internal
representation actually separates the three classes - independent of
whether its final guess happened to be right or wrong on those particular
three events. You'd expect the Hbb and QCD fingerprints to be less similar
to each other than, say, Hbb and Hcc - since Hbb and Hcc share a common
physics origin (both come from a Higgs boson) that QCD doesn't.

## Quick recap
- `model.eval()` + `torch.no_grad()` + the held-out test set gives you an honest accuracy score.
- A confusion matrix shows *which* classes get mixed up with which - not just an overall score.
- ROC curves and AUC summarize the tradeoff between catching real signal and letting background through, at every possible confidence threshold.
- You can peek at the model's internal 64-number "fingerprint" for any event to sanity-check that it's genuinely separating the three classes internally, not just getting lucky on final guesses.

That's the whole pipeline, start to finish - from raw CMS Open Data files
to a trained, evaluated transformer. Head back to the [main README](../README.md)
for how to actually run it end to end.

## Full code for this lesson

Copy this into your own Jupyter notebook cell(s), in order, as you go.

```python
model.eval()
correct = 0
total = 0

# Store predictions for a confusion matrix if you want to plot one later
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        outputs = model(batch_x)
        _, predicted = outputs.max(1)
        
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

test_acc = 100. * correct / total
print(f"Final Test Accuracy: {test_acc:.2f}%")
```

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F

model.eval()

all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        outputs = model(batch_x)
        
        # Apply softmax to get probabilities across the 3 classes
        probs = F.softmax(outputs, dim=1) 
        _, predicted = outputs.max(1)
        
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

# Convert lists to NumPy arrays for easier slicing
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# Generate the matrix
cm = confusion_matrix(all_labels, all_preds)

# Plotting
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Hbb (0)', 'Hcc (1)', 'QCD (2)'], 
            yticklabels=['Hbb (0)', 'Hcc (1)', 'QCD (2)'])

plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
plt.ylabel('True Class', fontsize=12, fontweight='bold')
plt.title('miniParT Confusion Matrix', fontsize=14)
plt.show()
```

```python
# Binarize the labels for One-vs-Rest ROC computation
# This turns a label like '1' into [0, 1, 0]
y_test_bin = label_binarize(all_labels, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green']
class_names = ['Hbb', 'Hcc', 'QCD']

# Calculate and plot ROC for each class
for i, color, name in zip(range(n_classes), colors, class_names):
    # fpr = False Positive Rate, tpr = True Positive Rate
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color=color, lw=2, 
             label=f'{name} vs Rest (AUC = {roc_auc:.3f})')

# Plot the random guessing baseline
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guessing')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Background Efficiency)', fontsize=12)
plt.ylabel('True Positive Rate (Signal Efficiency)', fontsize=12)
plt.title('miniParT ROC Curves', fontsize=14)
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.show()
```

```python
import torch.nn.functional as F
import pandas as pd

model.eval()

# 1. Grab one event from each class from our scaled test data
hbb_event, hcc_event, qcd_event = None, None, None

for i in range(len(y_test)):
    if y_test[i] == 0 and hbb_event is None:
        hbb_event = torch.tensor(X_test_scaled[i:i+1], dtype=torch.float32).to(device)
    elif y_test[i] == 1 and hcc_event is None:
        hcc_event = torch.tensor(X_test_scaled[i:i+1], dtype=torch.float32).to(device)
    elif y_test[i] == 2 and qcd_event is None:
        qcd_event = torch.tensor(X_test_scaled[i:i+1], dtype=torch.float32).to(device)
        
    if hbb_event is not None and hcc_event is not None and qcd_event is not None:
        break

# 2. Define a helper function to bypass the final MLP and get the 64D vector
def get_fingerprint(event_tensor):
    with torch.no_grad():
        # Project into 64D
        emb = model.embedding(event_tensor)
        # Pass through Self-Attention
        contextualized = model.transformer(emb)
        # Pool to get the single event-level fingerprint
        fingerprint = contextualized.mean(dim=1) 
    return fingerprint

# 3. Extract the fingerprints
fp_hbb = get_fingerprint(hbb_event)
fp_hcc = get_fingerprint(hcc_event)
fp_qcd = get_fingerprint(qcd_event)

# 4. Compute Cosine Similarities
# Cosine similarity bounds the dot product between -1 and 1
sim_hbb_hcc = F.cosine_similarity(fp_hbb, fp_hcc).item()
sim_hbb_qcd = F.cosine_similarity(fp_hbb, fp_qcd).item()
sim_hcc_qcd = F.cosine_similarity(fp_hcc, fp_qcd).item()

# 5. Display the results in a clean table
print("Cosine Similarity Matrix (1 = Identical, -1 = Opposite):\n")

data = {
    "Hbb": [1.0, sim_hbb_hcc, sim_hbb_qcd],
    "Hcc": [sim_hbb_hcc, 1.0, sim_hcc_qcd],
    "QCD": [sim_hbb_qcd, sim_hcc_qcd, 1.0]
}

df_sim = pd.DataFrame(data, index=["Hbb", "Hcc", "QCD"])
print(df_sim.round(3))
```
