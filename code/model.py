"""
model.py
========
The MiniParT model itself: a tiny transformer that looks at a pair of
jets and predicts Hbb / Hcc / QCD. See lessons/04_building_mini_part.md
for the full piece-by-piece explanation.
"""

import torch.nn as nn


class MiniParT(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4, hidden_dim=128, num_classes=3):
        super(MiniParT, self).__init__()

        # 1. Embedding: translate each jet's raw features into a richer
        #    internal description (input_dim -> embed_dim numbers)
        self.embedding = nn.Linear(input_dim, embed_dim)

        # 2. Self-attention: let the two jets exchange information about
        #    each other, using num_heads parallel "viewpoints", stacked
        #    2 layers deep so information can be exchanged twice.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
            dropout=0.1,
        )
        # Using just 2 layers for a "mini" model to keep local training fast
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 3. Classification head: turn the pooled event summary into 3
        #    final class scores
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len=2, Features)

        x = self.embedding(x)          # -> (Batch, 2, embed_dim)
        x = self.transformer(x)        # -> (Batch, 2, embed_dim)

        # Mean pooling over the 2 jets -> one summary per event
        x_pooled = x.mean(dim=1)       # -> (Batch, embed_dim)

        out = self.mlp(x_pooled)       # -> (Batch, num_classes)
        return out

    def get_fingerprint(self, x):
        """
        Returns the pooled embed_dim-length internal representation of
        an event -- the model's "fingerprint" of it -- bypassing the
        final classification head. See the "Bonus" section of
        lessons/06_evaluating_the_model.md.
        """
        emb = self.embedding(x)
        contextualized = self.transformer(emb)
        return contextualized.mean(dim=1)
