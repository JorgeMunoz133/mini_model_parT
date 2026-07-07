# Lesson 4 — Building MiniParT

This is the heart of the project. We're going to go through
`class MiniParT` in `code/model.py` piece by piece. Nothing here is magic
— every line is one of a handful of building blocks stacked together.

```python
class MiniParT(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4, hidden_dim=128, num_classes=3):
```

First, the knobs (these are called **hyperparameters** — settings *we*
choose, as opposed to the numbers the model learns on its own):

- **`input_dim`** — how many numbers describe one jet. We pass in
  `len(FEATURE_NAMES) = 10` (Lesson 1).
- **`embed_dim=64`** — the size of the model's own "internal language."
  More on this in a moment.
- **`num_heads=4`** — how many independent "attention" viewpoints the
  model uses at once. More below.
- **`hidden_dim=128`** — the size of an internal working layer used both
  inside the transformer and in the final decision-making step.
- **`num_classes=3`** — how many possible answers there are (Hbb, Hcc,
  QCD).

## Piece 1 — The embedding layer: translating raw numbers into a richer language

```python
self.embedding = nn.Linear(input_dim, embed_dim)
```

Each jet arrives as just 10 plain numbers. That's a very cramped way to
represent something as complex as a jet. `nn.Linear(10, 64)` is a simple,
learnable transformation that takes those 10 numbers and re-expresses them
as 64 numbers instead — not by adding new information, but by learning a
useful *combination* of the original 10 that's more expressive to work
with internally. Think of it like a translator: it doesn't know more facts
than the original sentence, but it can phrase things in a way that's
easier to reason about downstream. This step is called an **embedding**.

## Piece 2 — Self-attention: letting the two jets "talk" to each other

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=embed_dim,
    nhead=num_heads,
    dim_feedforward=hidden_dim,
    batch_first=True,
    dropout=0.1,
)
self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
```

This is the "Transformer" in MiniParT. Its core trick is called
**self-attention**: every jet in the pair gets to look at every other jet
(including itself) and decide *how much to pay attention to it* before
updating its own internal description.

Here, with only 2 jets, you can think of it very literally: jet A "looks
at" jet B and asks "given what you look like, how should I adjust what I
think I am?" — and jet B does the same, looking back at jet A. This
matters physically: whether a jet pair is Hbb, Hcc, or QCD isn't just
about what one jet looks like in isolation, it's about the *relationship*
between the two jets. Self-attention is what lets the model reason about
jets *together* instead of scoring each one separately and averaging at
the end.

A few of the settings:

- **`nhead=num_heads=4`** — instead of computing attention just one way,
  the model computes it 4 different ways in parallel, called **attention
  heads**. Think of 4 different reviewers looking at the same pair of
  jets, each one free to pick up on a different kind of relationship
  between them — then their observations get combined. This gives the
  model more than one "lens" to look through.
- **`dim_feedforward=hidden_dim=128`** — inside each transformer layer,
  after attention, there's a small additional processing step (another
  linear layer) that gets to further refine each jet's description; `128`
  is how wide that internal step is.
- **`dropout=0.1`** — during training, this randomly "switches off" 10% of
  the internal connections on every pass. That sounds destructive, but
  it's actually a safeguard: it forces the model to not over-rely on any
  single connection, which helps it generalize instead of just
  memorizing. (Dropout is automatically turned off when the model is just
  making predictions, not training.)
- **`num_layers=2`** — we stack **two** of these self-attention layers.
  After the first layer, each jet's description already includes some
  information "borrowed" from the other jet. Running a second layer lets
  that refined information get exchanged and refined *again* — like two
  rounds of a conversation instead of one.

## Piece 3 — Mean pooling: turning "2 jets" into "1 decision"

```python
x_pooled = x.mean(dim=1)   # shape: (Batch, embed_dim)
```

After the transformer, we still have a separate description for each of
the 2 jets. But we ultimately need one answer per *event* (Hbb/Hcc/QCD),
not one per jet. **Mean pooling** simply averages the two jets' 64-number
descriptions together into a single 64-number summary of the whole pair.
It's a deliberately simple way to combine them — more sophisticated models
use fancier combination methods, but averaging is easy to understand and
works fine for a "mini" model.

## Piece 4 — The classification head: making the final call

```python
self.mlp = nn.Sequential(
    nn.Linear(embed_dim, hidden_dim),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(hidden_dim, num_classes),
)
```

This is a small standard neural network (a "Multi-Layer Perceptron," or
MLP) that takes the pooled 64-number summary and turns it into 3 numbers —
one raw score per class (Hbb, Hcc, QCD). Higher score means "the model
thinks this class is more likely."

- `nn.Linear(64, 128)` expands the summary into a wider working space.
- `nn.ReLU()` is an **activation function** — it just zeroes out any
  negative number and leaves positive numbers unchanged. Without
  something like this, stacking linear layers on top of each other would
  mathematically collapse into being just one linear layer no matter how
  many you stack — ReLU is what lets the network learn genuinely
  non-linear, more complex patterns.
- `nn.Dropout(0.1)` — same idea as before, another safeguard against
  over-memorizing.
- `nn.Linear(128, 3)` — the final layer, producing exactly 3 numbers, one
  per class.

## Putting it together: the forward pass

"Forward pass" just means: here's how data actually flows through the
model, start to finish.

```python
def forward(self, x):
    # x shape: (Batch, 2, 10)   <- a batch of jet pairs, 10 features each
    x = self.embedding(x)        # -> (Batch, 2, 64)  translate each jet
    x = self.transformer(x)      # -> (Batch, 2, 64)  jets exchange info
    x_pooled = x.mean(dim=1)     # -> (Batch, 64)     average the 2 jets
    out = self.mlp(x_pooled)     # -> (Batch, 3)      final class scores
    return out
```

Reading the shape comments is the easiest way to keep track of what's
happening: we start with 2 jets described by 10 raw numbers each, expand
each jet's description to 64 richer numbers, let the two jets exchange
information twice, average them into one 64-number event summary, and
finally boil that down to 3 scores — one per possible answer.

## Quick recap
- The embedding layer translates each jet's 10 raw numbers into a richer 64-number internal description.
- Self-attention lets the two jets exchange information about each other — using 4 parallel "attention heads," repeated over 2 stacked layers.
- Mean pooling merges the two jets' descriptions into one summary per event.
- A small MLP turns that summary into 3 final class scores.
- Next: [Lesson 5 — how the model actually learns from examples](05_training_the_model.md)

## Full code for this lesson

Copy this into your own Jupyter notebook cell(s), in order, as you go.

```python
class MiniParT(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4, hidden_dim=128, num_classes=3):
        super(MiniParT, self).__init__()
        
        # 1. Linear projection (Embedding)
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # 2. Transformer Encoder Layer (Self-Attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            batch_first=True,
            dropout=0.1
        )
        # Using just 2 layers for a "mini" model to keep local training fast
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 3. Classification Head
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len=2, Features)
        
        # Project features
        x = self.embedding(x) # shape: (Batch, 2, embed_dim)
        
        # Apply self-attention
        x = self.transformer(x) # shape: (Batch, 2, embed_dim)
        
        # Mean pooling over the sequence (the 2 jets)
        x_pooled = x.mean(dim=1) # shape: (Batch, embed_dim)
        
        # Classify
        out = self.mlp(x_pooled) # shape: (Batch, num_classes)
        return out

model = MiniParT(input_dim=len(FEATURE_NAMES))
print(model)
```
