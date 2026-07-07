# Getting the data

This project needs three CMS Open Data files. They are **not included** in
this repo (each dataset is many gigabytes) — you need to download them
yourself and place them here as:

```
datasets/
├── ttHTobb.root
├── ttHTocc.root
└── qcd_bctoe.root
```

## Where each file comes from

| File you need | CERN Open Data record | What it is |
|---|---|---|
| `ttHTobb.root` | [record 67645](https://opendata.cern.ch/record/67645) | Simulated collisions where a Higgs boson decays to two bottom quarks (H→bb) |
| `ttHTocc.root` | [record 67651](https://opendata.cern.ch/record/67651) | Simulated collisions where a Higgs boson decays to two charm quarks (H→cc) |
| `qcd_bctoe.root` | [record 63242](https://opendata.cern.ch/record/63242) | Simulated ordinary background collisions (QCD), no Higgs boson involved |

All three are **NanoAODSIM** format — simulated data that mimics real 2016
CMS collisions, released under a [CC0 public domain license](https://creativecommons.org/publicdomain/zero/1.0/).
"Simulated" doesn't mean fake — it means physicists used the known laws of
particle physics to generate what a real collision of that type *would* look
like passing through the CMS detector. This is standard practice: you need
simulated, truth-labeled data like this to train (and to test) any
classifier, because for real collision data nobody hands you an answer key.

## How to download

Each record page has a "Files and indexes" section with individual `.root`
file links, and instructions for streaming via XRootD instead of a full
download (useful since a single file can be gigabytes). See:

- [Getting started with CMS NanoAOD](https://opendata.cern.ch/docs/cms-getting-started-nanoaod)
- [Using Docker containers for CMS Open Data](https://opendata.cern.ch/docs/cms-guide-docker)

You don't need every file listed on a record page — one or two files from
each dataset is plenty for MiniParT, since `code/features.py` lets you cap
the number of events read with `max_events`.

## A note on file names

The code in this repo expects exactly these three names inside `datasets/`.
If you download files with different names (CMS files usually have long
names describing the full production chain), just rename them, or edit the
paths at the top of `code/run_all.py`.
