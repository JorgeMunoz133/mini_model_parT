# Lesson 5 — Training the Model

The model from Lesson 4 starts out knowing nothing — its internal numbers
("weights") are randomly initialized. Training is the repeated process of
showing it examples, checking how wrong its guesses are, and nudging its
weights to be a little less wrong next time. This all lives in
`code/train.py`.

## Setup

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

- **`device`** — trains on a GPU (`cuda`) if one is available, since GPUs
  do this kind of math much faster than a CPU; otherwise it falls back to
  the CPU. Either works for a model this small, just at different speeds.
- **`criterion` (the loss function)** — `CrossEntropyLoss` is how we
  measure "how wrong was the guess." It compares the model's 3 class
  scores against the true label and produces a single number: low if the
  model was confidently correct, high if it was confidently wrong. Think
  of it as an automatic grader.
- **`optimizer`** — `AdamW` is the algorithm that actually adjusts the
  model's internal weights based on the loss. Think of it as the coach:
  after seeing the grade, it decides how to nudge every single tunable
  number in the model to make it do a little better next time.
  - **`lr=1e-3`** (learning rate) — how big a nudge to make each step.
    Too big, and the model overshoots and never settles down. Too small,
    and it learns painfully slowly. `0.001` is a common, reasonable
    starting point.
  - **`weight_decay=1e-4`** — a mild extra pressure that discourages the
    model's weights from growing unnecessarily large. Like dropout, this
    is another safeguard against over-memorizing the training examples
    instead of learning the general pattern.

## The training loop

```python
epochs = 10
for epoch in range(epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()
```

One **epoch** is one full pass through every batch in the training set.
We do 10 of them (`epochs = 10`). Inside each epoch, for every batch of
256 jet pairs:

1. **`model.train()`** — tells the model "we're training right now," which
   switches on things like dropout (Lesson 4) that should only be active
   during training.
2. **`optimizer.zero_grad()`** — clears out any leftover "how should each
   weight change" information from the previous batch. Without this, the
   nudges from different batches would incorrectly pile on top of each
   other.
3. **`outputs = model(batch_x)`** — the forward pass from Lesson 4: run
   this batch of jet pairs through the model and get back 3 class scores
   for each one.
4. **`loss = criterion(outputs, batch_y)`** — the automatic grader
   compares those scores to the true labels and produces one number: how
   wrong was this batch of guesses, on average.
5. **`loss.backward()`** — this is where PyTorch does something clever
   called **backpropagation**: working backward through every layer of
   the model, it calculates exactly how much *each individual weight*
   contributed to the error, so we know which direction to nudge each
   one.
6. **`optimizer.step()`** — the coach actually applies those nudges,
   updating every weight in the model by a small amount in the direction
   that should reduce the loss.

Repeat that for every batch, for 10 epochs, and the model gradually gets
better at telling Hbb, Hcc, and QCD jet pairs apart.

## Watching it learn

```python
train_acc = 100. * correct / total
print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
```

After each epoch, we print the average loss and the training accuracy (the
percentage of training examples the model got right *during* that epoch,
before the latest updates fully take effect). You should generally see the
loss go down and accuracy go up epoch over epoch — that's the model
visibly learning. If accuracy stalls or loss starts climbing back up, that
can be a sign of problems worth investigating (too high a learning rate,
not enough data, etc.) — but working through that is beyond MiniParT's
scope as a teaching example.

Note this is *training* accuracy, measured on data the model has already
seen — it's useful to sanity-check learning is happening, but it's not a
fair report card. For that, we need data the model has never seen at all,
which is exactly what the next lesson covers.

## Quick recap
- `CrossEntropyLoss` grades how wrong the model's guesses are; `AdamW` decides how to adjust the model's weights in response.
- Each batch: clear old gradients → forward pass → compute loss → backpropagate → update weights.
- One full pass through all batches is an epoch; we repeat for several epochs so the model keeps improving.
- Training accuracy is a useful sanity check, but not a fair test — see [Lesson 6](06_evaluating_the_model.md) for that.

## Full code for this lesson

Copy this into your own Jupyter notebook cell(s), in order, as you go.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()
        
    train_acc = 100. * correct / total
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
```
