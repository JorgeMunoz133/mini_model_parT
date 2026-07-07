# Lesson 1 — What Is a Jet?

## The detector, in one picture (with words)

Picture the CMS detector as a giant cylindrical onion, wrapped around the
pipe where protons collide. When a collision happens in the middle, quarks
and other particles shoot outward in all directions. A single quark can't
exist alone in nature for long — as it flies outward, it drags a cloud of
other particles along with it, like a rock thrown into wet sand kicking up
a spray of grains. **That whole spray, moving roughly together in one
direction, is called a jet.**

The detector doesn't record "a bottom quark went this way." It records
where dozens of individual particles in that spray landed and how much
energy each one carried. Physicists then run reconstruction software that
groups those particles back together into what we call a jet, and computes
some useful summary numbers about it. **Those summary numbers are what our
model actually sees** — not the raw particle hits.

## The 10 numbers we give the model

For every jet, we use these 10 features (this list lives in `code/features.py`
as `FEATURE_NAMES`):

### Where the jet is and how big it is

- **`Jet_pt`** (transverse momentum) — how hard the jet is moving,
  measured sideways relative to the beam pipe, in GeV. Bigger number =
  more energetic jet. Think of it as "how fast and heavy is this spray of
  particles moving away from the middle."

- **`Jet_eta`** (pseudorapidity) — describes the jet's angle relative to
  the beam pipe. `eta = 0` means the jet went straight out sideways;
  large positive or negative values mean it went almost straight down the
  beam pipe direction. Think of it as "how close to the beam line did this
  spray go."

- **`Jet_phi`** — the jet's angle *around* the beam pipe, like a compass
  direction (0 to 2π radians) as you look straight down the pipe.
  `Jet_eta` and `Jet_phi` together pin down exactly which direction the
  jet flew, the same way latitude and longitude pin down a spot on Earth.

- **`Jet_mass`** — the combined mass of everything in the jet, worked out
  from the energy and momentum of every particle inside it. A tighter,
  simpler jet tends to have lower mass than a jet that's really two things
  overlapping.

### What the jet is made of

Every particle in the spray is either **charged** or **neutral**, and is
either a **hadron** (built from quarks, like a proton or a pion) or
behaves like **electromagnetic** radiation (electrons and photons). These
next four numbers ("energy fractions") say what *fraction* of the jet's
total energy falls into each of those four buckets. They always add up to
roughly 1 (100%):

- **`Jet_chHEF`** — Charged Hadron Energy Fraction. What fraction of the
  jet's energy comes from charged hadron-type particles.
- **`Jet_neHEF`** — Neutral Hadron Energy Fraction. Same idea, for neutral
  hadron-type particles.
- **`Jet_chEmEF`** — Charged Electromagnetic Energy Fraction (mostly
  electrons/positrons in the jet).
- **`Jet_neEmEF`** — Neutral Electromagnetic Energy Fraction (mostly
  photons in the jet).

Why do these help? Different quark types tend to "hadronize" (turn into a
jet) in slightly different ways — for example, jets from heavier quarks
are statistically a little more likely to contain certain kinds of
particles than jets from ordinary background processes. No single fraction
gives it away, but the combination is a real clue.

### How the jet is put together

- **`Jet_nConstituents`** — simply, how many individual particles were
  found inside the jet. A jet built from more sub-pieces "looks" different
  to the model than a tight jet built from just a few particles.

- **`Jet_puId`** — Pileup ID. At the LHC, several unrelated proton
  collisions actually happen at almost the same instant inside the
  detector ("pileup"). This is a score estimating how likely the jet is to
  be a genuine jet from *our* interesting collision, versus stray junk
  left over from one of those other unrelated, boring collisions. Higher
  is usually more "real."

## Why not just look at raw particles?

The real, full-size Particle Transformer does exactly that — it looks at
every individual particle inside a jet (there can be dozens), not just 10
summary numbers. That's more powerful but much bigger and slower to train.
MiniParT uses these 10 pre-computed summary numbers per jet instead, which
is why it can train on a laptop instead of a GPU cluster — the tradeoff is
some detail gets thrown away.

## Where this data lives

These numbers come straight out of **NanoAOD** files — CMS's compact,
public data format, stored as [ROOT](https://root.cern.ch/) files. We read
them in Python using the `uproot` library, which lets you pull out a
"branch" (a column of data) by name without needing any CERN-specific
software installed. You'll see this in `code/features.py`:

```python
import uproot
tree = uproot.open("datasets/ttHTobb.root")["Events"]
events = tree.arrays(FEATURE_NAMES, entry_stop=max_events)
```

`tree.arrays(...)` reads the requested columns for every event (a
collision) into memory. `entry_stop=max_events` just caps how many events
to read, so you can test quickly on a small slice before running on
everything.

## Quick recap
- A jet is a spray of particles created when a quark flies out of a collision.
- We describe each jet with 10 numbers: 4 about its size/direction (`pt`, `eta`, `phi`, `mass`), 4 about what it's made of (the energy fractions), and 2 about its structure (`nConstituents`, `puId`).
- These numbers are read from public CMS NanoAOD files using `uproot`.
- Next: [Lesson 2 — how do we know the *right answer* for each jet during training?](02_finding_the_truth_labels.md)
