# Lesson 2 - Finding the Truth Labels

## Why we need an answer key

To train a model with examples, we need to already know the right answer
for each example - otherwise there's nothing to learn from. This is called
**supervised learning**: "supervised" because a known answer supervises
(corrects) the model while it learns.

For real collision data, nobody can look at the debris and just *know*
which quark caused which jet - that information isn't directly visible.
But for **simulated** data, it's different: the simulation software
generated the whole event starting from "let's create a Higgs boson that
decays to two bottom quarks," so it also secretly records what actually
happened at the truth level, before detector effects blur things. That
secret record is stored as extra columns starting with `GenPart_*`
("Generator-level Particle"). We only use these truth columns to *build
our training labels* - the model itself never sees them.

## The two signal files (ttHTobb, ttHTocc): matching jets to truth

For the signal samples, we need to figure out: *of all the jets in this
event, which ones actually came from the Higgs boson's b-quarks (or
c-quarks)?* This is a two-step process, all handled inside
`extract_features()` in `code/features.py`.

### Step 1 - Find the Higgs boson's daughter quarks

Every particle in `GenPart_*` has:
- `GenPart_pdgId` - an ID number identifying *what* the particle is.
  Physicists use a standard numbering scheme called the **PDG ID**: `5`
  means a bottom quark, `4` means a charm quark (and `-5`/`-4` are their
  antimatter partners, which is why we compare `abs(pdgId)`), `25` means
  a Higgs boson.
- `GenPart_genPartIdxMother` - which earlier particle in the list is this
  particle's "parent" (the thing that decayed into it).

So "find the Higgs boson's daughter quarks" becomes: *find particles
whose ID is ±5 (or ±4) and whose parent's ID is 25.*

```python
target_pdg = 5 if label == 0 else 4          # 5 = bottom quark, 4 = charm quark
mother_pdg = ak.where(valid, events.GenPart_pdgId[mother_idx], -999)
is_higgs_dau = (abs(events.GenPart_pdgId) == target_pdg) & (mother_pdg == 25)
```

A Higgs decaying to two quarks should have exactly two of these, so we
also throw away any event where that isn't true:

```python
mask = ak.num(events.GenPart_pt[is_higgs_dau]) == 2
```

### Step 2 - Match those quarks to actual reconstructed jets

Knowing *which truth-level quarks* came from the Higgs boson isn't quite
enough - we need to know *which of the actual reconstructed jets* in the
detector correspond to them. We do that with a geometric trick.

Remember `eta` and `phi` from Lesson 1 - they're like latitude and
longitude for a particle's direction. We can measure the "distance" on
that map between a jet and a truth quark using a quantity called **ΔR**
(read "delta R"):

```
ΔR = sqrt( (Δeta)² + (Δphi)² )
```

If ΔR is small, the jet and the quark are pointing in almost the same
direction - good evidence that jet *is* the spray created by that quark.
We use a common threshold of **ΔR < 0.4**:

```python
dr1 = np.sqrt((jets.eta - d1.eta[:, None])**2 + delta_phi(jets.phi, d1.phi[:, None])**2)
matched = (dr1 < 0.4) | (dr2 < 0.4)
```

### Why `delta_phi` needs its own function

`phi` is an angle that wraps around a circle (0 to 2π, then back to 0) -
like a clock face. If one jet is at `phi = 0.1` and another is at
`phi = 6.2` (close to 2π), a naive subtraction says they're almost
3.5 radians apart, when really they're neighbors, just on either side of
the "12 o'clock" wraparound point. `delta_phi` fixes that:

```python
def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    return (dphi + np.pi) % (2*np.pi) - np.pi
```

This squeezes the difference back into the range `[-π, π]`, so it always
reports the *shortest* way around the circle - exactly like saying "11
o'clock to 1 o'clock is 2 hours apart," not 10.

### Keeping exactly two matched jets

Finally, we only keep events where exactly two jets matched - one for each
Higgs daughter quark - since MiniParT is built to always look at a pair of
jets:

```python
mask_2jets = ak.num(matched_events.Jet_pt) == 2
final_events = matched_events[mask_2jets]
```

## The background file (QCD): no matching needed

For the QCD background sample, there's no Higgs boson to match to at all
- by definition, it's not there. So we take a simpler approach: just grab
the two highest-momentum ("leading") jets in each event, since those are
the ones most likely to matter for a physics analysis:

```python
mask_2jets = ak.num(events.Jet_pt) >= 2
events = events[mask_2jets]
final_events = events[:, :2]     # keep the first 2 jets
```

## Turning this into labels

Every event that survives becomes one training example: 2 jets × 10
features, with one label attached:

- `label = 0` → the jet pair is from **Hbb**
- `label = 1` → the jet pair is from **Hcc**
- `label = 2` → the jet pair is **QCD** background

```python
X = np.stack(feature_list, axis=-1)   # shape: (n_events, 2, 10)
y = np.full(X.shape[0], label)
```

## Quick recap
- Truth-level (`GenPart_*`) columns exist only in simulation, and only get used to *build labels* - never fed to the model.
- We find the Higgs boson's daughter quarks by PDG ID (5 = bottom, 4 = charm) and mother ID (25 = Higgs).
- We match those truth quarks to real jets using ΔR - a "distance on the sky" built from `eta` and `phi`.
- QCD background just uses the two leading jets, since there's no Higgs decay to match to.
- Next: [Lesson 3 - preparing this data to actually feed into a neural network](03_preparing_the_data.md)

## Full code for this lesson

Copy this into your own Jupyter notebook cell(s), in order, as you go.

```python
def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    return (dphi + np.pi) % (2*np.pi) - np.pi

def extract_features(filepath, label, is_signal=True, max_events=None):
    tree = uproot.open(filepath)["Events"]
    
    # Load required branches
    branches = FEATURE_NAMES.copy()
    if is_signal:
        branches += [
            "GenPart_pdgId", "GenPart_pt", "GenPart_eta", 
            "GenPart_phi", "GenPart_mass", "GenPart_genPartIdxMother"
        ]
    
    events = tree.arrays(branches, entry_stop=max_events)
    
    if is_signal:
        # Determine target quark based on label (0: b-quark=5, 1: c-quark=4)
        target_pdg = 5 if label == 0 else 4
        
        mother_idx = events.GenPart_genPartIdxMother
        valid = mother_idx >= 0
        mother_pdg = ak.where(valid, events.GenPart_pdgId[mother_idx], -999)
        
        is_higgs_dau = (abs(events.GenPart_pdgId) == target_pdg) & (mother_pdg == 25)
        mask = ak.num(events.GenPart_pt[is_higgs_dau]) == 2
        events = events[mask]
        is_higgs_dau = is_higgs_dau[mask]
        
        # Build 4-vectors
        jets = ak.zip({
            "pt": events.Jet_pt, "eta": events.Jet_eta,
            "phi": events.Jet_phi, "mass": events.Jet_mass
        }, with_name="Momentum4D")
        
        dau = ak.zip({
            "pt": events.GenPart_pt[is_higgs_dau], "eta": events.GenPart_eta[is_higgs_dau],
            "phi": events.GenPart_phi[is_higgs_dau], "mass": events.GenPart_mass[is_higgs_dau]
        }, with_name="Momentum4D")
        
        d1, d2 = dau[:, 0], dau[:, 1]
        
        # Match using dR < 0.4
        dr1 = np.sqrt((jets.eta - d1.eta[:, None])**2 + delta_phi(jets.phi, d1.phi[:, None])**2)
        dr2 = np.sqrt((jets.eta - d2.eta[:, None])**2 + delta_phi(jets.phi, d2.phi[:, None])**2)
        matched = (dr1 < 0.4) | (dr2 < 0.4)
        
        # Extract features for matched jets
        matched_events = events[matched]
        
        # Keep exactly 2 matched jets
        mask_2jets = ak.num(matched_events.Jet_pt) == 2
        final_events = matched_events[mask_2jets]
        
    else:
        # For QCD, require at least 2 jets and take the top 2 leading jets
        mask_2jets = ak.num(events.Jet_pt) >= 2
        events = events[mask_2jets]
        # Slice to keep only the first 2 jets
        final_events = events[:, :2]

    # Stack features into a NumPy array of shape (N_events, 2_jets, N_features)
    feature_list = []
    for feat in FEATURE_NAMES:
        # Fill missing values with 0 (e.g., puId might have NaNs depending on pt)
        arr = ak.fill_none(final_events[feat], 0)
        feature_list.append(ak.to_numpy(arr))
        
    X = np.stack(feature_list, axis=-1)
    y = np.full(X.shape[0], label)
    
    print(f"Loaded label {label}: {X.shape[0]} events")
    return X, y
```
