# Lesson 6 — Evaluating the Model

Training accuracy (Lesson 5) can lie to you — a model can look great on
data it's already memorized and still be useless on new data. This lesson
is about actually finding out whether MiniParT learned something real,
using only the 20% of data it never saw during training. All of this is in
`code/evaluate.py`.

## Step 1 — Test accuracy

```python
model.eval()
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        _, predicted = outputs.max(1)
        correct += predicted.eq(batch_y).sum().item()

test_acc = 100. * correct / total
```

- **`model.eval()`** — the opposite of `model.train()` from Lesson 5. It
  switches off dropout, so the model always gives its single best,
  consistent answer instead of the slightly-randomized version used
  during training.
- **`torch.no_grad()`** — tells PyTorch not to bother tracking how to
  compute gradients here, since we're not training, just checking
  answers. This makes evaluation faster and uses less memory.
- **`outputs.max(1)`** — remember the model outputs 3 raw scores per
  event (Lesson 4). `.max(1)` just picks out whichever of the 3 is
  highest — that becomes the model's actual predicted class.
- **`test_acc`** — the percentage of *unseen* test examples the model
  classified correctly. This is the honest report card.

## Step 2 — The confusion matrix: not just "right or wrong," but *how* wrong

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
the two Higgs decay types specifically — which, given how physically
similar bottom and charm jets are, is exactly where you'd expect it to
struggle most.

## Step 3 — ROC curves: how good is each class at different confidence thresholds

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

- **True Positive Rate** (y-axis) — of all the real Hbb events, what
  fraction did we correctly flag as Hbb, at some confidence threshold?
  Physicists often call this "signal efficiency."
- **False Positive Rate** (x-axis) — of all the events that were *not*
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
else," "QCD vs. everything else") — that's called a **one-vs-rest**
approach, and it's why the labels get "binarized" first
(`label_binarize`) into three separate yes/no columns.

## Bonus — Peeking inside the model with "fingerprints"

There's one more genuinely fun thing in the notebook worth understanding.
Instead of only looking at the model's *final* answer, we can grab its
internal 64-number description of an event — right after self-attention
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
representation actually separates the three classes — independent of
whether its final guess happened to be right or wrong on those particular
three events. You'd expect the Hbb and QCD fingerprints to be less similar
to each other than, say, Hbb and Hcc — since Hbb and Hcc share a common
physics origin (both come from a Higgs boson) that QCD doesn't.

## Quick recap
- `model.eval()` + `torch.no_grad()` + the held-out test set gives you an honest accuracy score.
- A confusion matrix shows *which* classes get mixed up with which — not just an overall score.
- ROC curves and AUC summarize the tradeoff between catching real signal and letting background through, at every possible confidence threshold.
- You can peek at the model's internal 64-number "fingerprint" for any event to sanity-check that it's genuinely separating the three classes internally, not just getting lucky on final guesses.

That's the whole pipeline, start to finish — from raw CMS Open Data files
to a trained, evaluated transformer. Head back to the [main README](../README.md)
for how to actually run it end to end.
