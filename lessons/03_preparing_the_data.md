# Lesson 3 — Preparing the Data

Once `code/features.py` and `code/labels.py` have turned all three files
into arrays of jet pairs and labels, we still can't hand this straight to
a neural network. Three things need to happen first — all in
`code/prepare_data.py`.

## Step 1 — Combine everything into one big pile

```python
X = np.concatenate([X_bb, X_cc, X_qcd], axis=0)
y = np.concatenate([y_bb, y_cc, y_qcd], axis=0)
```

`X` now holds every jet pair from all three datasets mixed together, shape
`(total_events, 2, 10)` — that's "however many events, 2 jets each, 10
numbers per jet." `y` holds the matching label (0, 1, or 2) for each one.

## Step 2 — Split into training data and test data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

We deliberately hold back 20% of the data (`test_size=0.2`) and never show
it to the model during training. Why? Because a model that's just
*memorized* its training examples would score perfectly on those examples
without having actually learned anything useful — like a student who
memorizes the answers to last year's exam instead of understanding the
material. Testing on data the model has genuinely never seen is the only
honest way to check whether it actually learned the underlying pattern.

- `random_state=42` just makes the "random" split repeatable — anyone
  running this code gets the exact same split, which makes results
  comparable and debugging easier.
- `stratify=y` makes sure the 80/20 split has the same *proportion* of
  Hbb, Hcc, and QCD examples in both the training set and the test set,
  instead of accidentally putting almost all the QCD examples in one side.

## Step 3 — Put every feature on the same "ruler" (scaling)

Look back at Lesson 1: `Jet_pt` might be a number like `85.0` (GeV), while
an energy fraction like `Jet_chHEF` is always between `0.0` and `1.0`.
If we feed those raw numbers straight into a neural network, the network
will initially treat the *size* of a number as if it were automatically
more important — `Jet_pt` would dominate simply because its values are
bigger, not because it's actually more useful.

The fix is **standardization**: for every feature, we shift and rescale
its values so that, across the whole training set, the average is `0` and
the spread (standard deviation) is `1`. Now every feature lives on the
same footing — think of it as converting inches and kilometers into the
same "how many steps away from average" scale before comparing them.

```python
scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, len(FEATURE_NAMES))     # (n_events*2, 10)
X_train_scaled = scaler.fit_transform(X_train_flat).reshape(-1, 2, len(FEATURE_NAMES))
X_test_scaled = scaler.transform(X_test_flat).reshape(-1, 2, len(FEATURE_NAMES))
```

Two important details:

- We `.reshape(-1, 10)` first because `StandardScaler` expects a plain
  table of rows and columns, not a 3D block — so we temporarily flatten
  "2 jets" into extra rows, scale, then reshape back.
- We call `.fit_transform()` on the **training** data (learn the average
  and spread *from* training data, then apply it), but only `.transform()`
  (never `.fit_transform()`) on the **test** data. The test set must be
  scaled using the training set's numbers, not its own — otherwise we'd be
  leaking a peek at the test data into how we prepared the training data,
  which quietly makes results look better than they really are.

## Step 4 — Turn everything into PyTorch tensors, in batches

Neural network libraries like PyTorch don't work directly on NumPy arrays
— they work on their own array type called a **tensor**, which supports
the extra bookkeeping needed for training (like automatically tracking how
to compute gradients — more in Lesson 5).

```python
train_data = TensorDataset(
    torch.tensor(X_train_scaled, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long),
)
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
```

Instead of showing the model all training examples at once (slow, and
memory-hungry) or one at a time (noisy, unstable), we show it small
handfuls at a time — **batches** of 256 examples. `DataLoader` handles
chopping the dataset into batches and, with `shuffle=True`, mixes up the
order every epoch (full pass through the data) so the model doesn't
accidentally learn something from the *order* the examples happen to be
stored in.

## Quick recap
- Combine all three datasets, then split 80/20 into train/test, keeping the class proportions equal on both sides (`stratify`).
- Scale every feature to the same "average 0, spread 1" footing, fitting the scaler only on training data.
- Convert to PyTorch tensors and feed the model small shuffled batches at a time, not the whole dataset at once.
- Next: [Lesson 4 — building the MiniParT model itself](04_building_mini_part.md)

## Full code for this lesson

Copy this into your own Jupyter notebook cell(s), in order, as you go.

```python
# Extract features (adjust max_events to None when ready for full training)
X_bb, y_bb = extract_features("datasets/ttHTobb.root", label=0, is_signal=True, max_events=50000)
X_cc, y_cc = extract_features("datasets/ttHTocc.root", label=1, is_signal=True, max_events=50000)
X_qcd, y_qcd = extract_features("datasets/qcd_bctoe.root", label=2, is_signal=False, max_events=50000)

# Combine datasets
X = np.concatenate([X_bb, X_cc, X_qcd], axis=0)
y = np.concatenate([y_bb, y_cc, y_qcd], axis=0)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize features (Transformers are sensitive to scale)
# We flatten to (N*2, Features) to fit the scaler, then reshape back
scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, len(FEATURE_NAMES))
X_test_flat = X_test.reshape(-1, len(FEATURE_NAMES))

X_train_scaled = scaler.fit_transform(X_train_flat).reshape(-1, 2, len(FEATURE_NAMES))
X_test_scaled = scaler.transform(X_test_flat).reshape(-1, 2, len(FEATURE_NAMES))

# Convert to PyTorch tensors
train_data = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_data = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
```
