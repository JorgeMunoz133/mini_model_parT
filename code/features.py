"""
features.py
============
Reads raw CMS Open Data (NanoAOD) files and turns them into training
examples: pairs of jets described by 10 numbers each, with a label
attached (0 = Hbb, 1 = Hcc, 2 = QCD).

See lessons/01_what_is_a_jet.md and lessons/02_finding_the_truth_labels.md
for the full plain-English explanation of every step here.
"""

import uproot
import awkward as ak
import vector
import numpy as np

vector.register_awkward()

# We avoid DeepJet/DeepCSV tagger variables on purpose -- those are
# themselves the output of other, more complex taggers, and using them
# would be "cheating": MiniParT should learn to separate Hbb/Hcc/QCD from
# more basic kinematic and energy-fraction information instead.
FEATURE_NAMES = [
    "Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass",
    "Jet_chHEF", "Jet_neHEF", "Jet_chEmEF", "Jet_neEmEF",
    "Jet_nConstituents", "Jet_puId",
]

# Labels used everywhere in this project: 0 = Hbb, 1 = Hcc, 2 = QCD


def delta_phi(phi1, phi2):
    """
    Angular difference between two phi angles, correctly wrapped around
    the 0-to-2*pi circle so it always reports the *shortest* way around.
    See lessons/02_finding_the_truth_labels.md for why this is needed.
    """
    dphi = phi1 - phi2
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def extract_features(filepath, label, is_signal=True, max_events=None):
    """
    Read one CMS NanoAOD file and return (X, y):
      X : NumPy array, shape (n_events, 2, len(FEATURE_NAMES))
          Two jets per event, 10 features per jet.
      y : NumPy array, shape (n_events,)
          The label (0, 1, or 2) repeated for every event.

    filepath   : path to a .root file, e.g. "datasets/ttHTobb.root"
    label      : 0 (Hbb), 1 (Hcc), or 2 (QCD)
    is_signal  : True for ttHTobb/ttHTocc (does truth-level jet matching),
                 False for QCD background (just takes the 2 leading jets)
    max_events : cap on how many events to read, useful for quick testing;
                 use None to read everything
    """
    tree = uproot.open(filepath)["Events"]

    branches = FEATURE_NAMES.copy()
    if is_signal:
        # Truth-level ("Generator Particle") columns -- only used to
        # figure out training labels, never fed to the model itself.
        branches += [
            "GenPart_pdgId", "GenPart_pt", "GenPart_eta",
            "GenPart_phi", "GenPart_mass", "GenPart_genPartIdxMother",
        ]

    events = tree.arrays(branches, entry_stop=max_events)

    if is_signal:
        # Step 1: find the Higgs boson's daughter quarks.
        # PDG ID: 5 = bottom quark, 4 = charm quark, 25 = Higgs boson.
        target_pdg = 5 if label == 0 else 4

        mother_idx = events.GenPart_genPartIdxMother
        valid = mother_idx >= 0
        mother_pdg = ak.where(valid, events.GenPart_pdgId[mother_idx], -999)

        is_higgs_dau = (abs(events.GenPart_pdgId) == target_pdg) & (mother_pdg == 25)
        mask = ak.num(events.GenPart_pt[is_higgs_dau]) == 2
        events = events[mask]
        is_higgs_dau = is_higgs_dau[mask]

        # Build 4-vectors for reconstructed jets and for the two truth quarks
        jets = ak.zip({
            "pt": events.Jet_pt, "eta": events.Jet_eta,
            "phi": events.Jet_phi, "mass": events.Jet_mass,
        }, with_name="Momentum4D")

        dau = ak.zip({
            "pt": events.GenPart_pt[is_higgs_dau], "eta": events.GenPart_eta[is_higgs_dau],
            "phi": events.GenPart_phi[is_higgs_dau], "mass": events.GenPart_mass[is_higgs_dau],
        }, with_name="Momentum4D")

        d1, d2 = dau[:, 0], dau[:, 1]

        # Step 2: match truth quarks to real jets using delta-R < 0.4
        dr1 = np.sqrt((jets.eta - d1.eta[:, None]) ** 2 + delta_phi(jets.phi, d1.phi[:, None]) ** 2)
        dr2 = np.sqrt((jets.eta - d2.eta[:, None]) ** 2 + delta_phi(jets.phi, d2.phi[:, None]) ** 2)
        matched = (dr1 < 0.4) | (dr2 < 0.4)

        matched_events = events[matched]

        # Keep only events where exactly 2 jets matched
        mask_2jets = ak.num(matched_events.Jet_pt) == 2
        final_events = matched_events[mask_2jets]

    else:
        # QCD background: no Higgs boson to match to, just take the
        # 2 leading (highest-momentum) jets in each event.
        mask_2jets = ak.num(events.Jet_pt) >= 2
        events = events[mask_2jets]
        final_events = events[:, :2]

    # Stack the 10 features into a NumPy array, shape (n_events, 2, 10)
    feature_list = []
    for feat in FEATURE_NAMES:
        arr = ak.fill_none(final_events[feat], 0)  # e.g. puId can be missing depending on pt
        feature_list.append(ak.to_numpy(arr))

    X = np.stack(feature_list, axis=-1)
    y = np.full(X.shape[0], label)

    print(f"Loaded label {label}: {X.shape[0]} events")
    return X, y
