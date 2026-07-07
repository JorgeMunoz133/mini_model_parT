"""
run_all.py
==========
Runs the whole MiniParT pipeline end to end: load data -> prepare data ->
build model -> train -> evaluate. This is the same sequence covered by
lessons 1 through 6 -- read those first if anything here is unclear.

Usage:
    cd code/
    python run_all.py
"""

from features import FEATURE_NAMES, extract_features
from prepare_data import prepare_data
from model import MiniParT
from train import train_model
from evaluate import evaluate_model, fingerprint_similarity

DATASETS_DIR = "../datasets"
MAX_EVENTS = 50_000  # set to None to use every event in each file


def main():
    # 1. Load & label each dataset (lessons 1-2)
    X_bb, y_bb = extract_features(f"{DATASETS_DIR}/ttHTobb.root", label=0, is_signal=True, max_events=MAX_EVENTS)
    X_cc, y_cc = extract_features(f"{DATASETS_DIR}/ttHTocc.root", label=1, is_signal=True, max_events=MAX_EVENTS)
    X_qcd, y_qcd = extract_features(f"{DATASETS_DIR}/qcd_bctoe.root", label=2, is_signal=False, max_events=MAX_EVENTS)

    # 2. Split / scale / batch (lesson 3)
    train_loader, test_loader, X_test_scaled, y_test, scaler = prepare_data(
        X_bb, y_bb, X_cc, y_cc, X_qcd, y_qcd
    )

    # 3. Build the model (lesson 4)
    model = MiniParT(input_dim=len(FEATURE_NAMES))
    print(model)

    # 4. Train (lesson 5)
    model, device = train_model(model, train_loader, epochs=10)

    # 5. Evaluate (lesson 6)
    evaluate_model(model, test_loader, device)
    fingerprint_similarity(model, X_test_scaled, y_test, device)


if __name__ == "__main__":
    main()
