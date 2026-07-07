# Lesson 0 — The Big Picture

## The question we're trying to answer

Deep inside the CMS detector at CERN, protons smash into each other at
nearly the speed of light. Most of the time, nothing interesting happens.
But every once in a while, that collision briefly creates a **Higgs boson**
— a special, heavy, short-lived particle — which almost instantly falls
apart ("decays") into other particles.

We're interested in one specific way it can fall apart:

- Sometimes the Higgs boson decays into **two bottom quarks** (physicists
  call this "**Hbb**", said "H to b b").
- Sometimes it decays into **two charm quarks** ("**Hcc**", "H to c c").
- And most of the time, whatever sprays of particles we see in the
  detector have **nothing to do with a Higgs boson at all** — they're just
  ordinary background collisions. Physicists call this catch-all category
  "**QCD**" background.

Quarks can't fly around on their own — as they shoot away from the
collision, they drag along a spray of other particles, like a firework
exploding. That spray is called a **jet**. So what we actually see in the
detector is jets, not quarks directly.

**Our job: look at a pair of jets and guess whether they came from Hbb,
Hcc, or QCD.**

This is hard for a very physical reason: bottom quarks and charm quarks are
cousins — both are "heavy" quarks that behave similarly when they turn into
jets. Telling their jets apart is a genuinely open, actively-researched
problem in particle physics. Telling either of them apart from ordinary QCD
jets is a bit easier, but still not trivial.

## Why use AI for this at all?

A human physicist can't look at raw detector data and just "see" whether a
jet came from a bottom quark, a charm quark, or nothing special. But each
jet does leave subtle clues — how much of its energy is in charged vs.
neutral particles, how spread out it is, how many particles it contains,
and so on. No single clue is a smoking gun, but *combined*, they carry real
information.

That's exactly the kind of problem machine learning is good at: finding a
pattern across many weak, noisy clues that no simple rule can capture.

## What is "machine learning," in one paragraph?

Instead of a person writing down rules like "if energy fraction > 0.6 then
it's probably a bottom quark," we show a computer program **thousands of
examples where we already know the right answer** (because the data is
simulated — see [Lesson 2](02_finding_the_truth_labels.md)), and let it
gradually adjust itself until it gets good at guessing correctly. That
process — adjusting itself based on examples — is "training," and the
program doing the adjusting is the "model."

## Why a *transformer*, and why "mini"?

A **transformer** is a type of model that's very good at looking at a
*set* of things and figuring out how they relate to each other — famously
used for understanding sentences (where words relate to other words), and
also used in real CMS physics analyses to look at a *set of jets* and
figure out how they relate to each other. We only have **two jets** per
event here, so it's a toy-sized version of the same idea used in
professional particle physics AI — which is exactly why it's called
**MiniParT** ("mini Particle Transformer").

It's "mini" because:
- It only looks at 2 jets at a time (real versions can handle 100+ particles per jet)
- It only uses 10 simple numbers per jet (real versions use many more, plus raw particle-level data)
- The network itself is small enough to train on a laptop in minutes, not hours

## Roadmap

1. [**What Is a Jet?**](01_what_is_a_jet.md) — the raw ingredients: 10 numbers per jet
2. [**Finding the Truth Labels**](02_finding_the_truth_labels.md) — how we know the "right answer" for training
3. [**Preparing the Data**](03_preparing_the_data.md) — getting the numbers ready for a neural network
4. [**Building MiniParT**](04_building_mini_part.md) — the model itself, piece by piece
5. [**Training the Model**](05_training_the_model.md) — how it actually learns
6. [**Evaluating the Model**](06_evaluating_the_model.md) — did it work, and how do we know?

## Quick recap
- We're teaching a computer to classify pairs of jets into Hbb / Hcc / QCD.
- We use simulated CMS data because it comes with a built-in answer key.
- MiniParT is a scaled-down version of a real particle physics AI architecture — small enough to fully understand, built the same way the real ones are.
