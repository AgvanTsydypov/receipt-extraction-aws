"""Train a confidence model that decides which documents can skip human review.

The answer for each document comes from the primary method. The model predicts whether
that answer is fully correct, using only signals available in production (no ground truth):
agreement with a second method, receipt arithmetic and Textract confidence scores.

The acceptance threshold is chosen with 5-fold cross-validation on the train split to reach
the target precision, then the policy is evaluated once on the test split.

Usage:
    python scripts/train_confidence.py
    python scripts/train_confidence.py --target full --precision 0.95
"""

import argparse
import json
from datetime import datetime

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from idp.config import RESULTS_DIR
from idp.features import build_features, is_correct, textract_signals
from idp.runs import find_run, load_records


def build_dataset(primary_sel: str, secondary_sel: str, split: str, target: str):
    primary = load_records(find_run(primary_sel, split))
    secondary = load_records(find_run(secondary_sel, split))
    ids = sorted(
        d for d in primary.keys() & secondary.keys()
        if not primary[d]["error"] and not secondary[d]["error"]
    )
    if not ids:
        raise SystemExit(f"No usable documents in split '{split}'")
    rows = [
        build_features(primary[d]["prediction"], secondary[d]["prediction"], textract_signals(split, d))
        for d in ids
    ]
    names = list(rows[0])
    X = np.array([[row[n] for n in names] for row in rows], dtype=float)
    y = np.array([is_correct(primary[d], target) for d in ids], dtype=int)
    return ids, names, X, y


def pick_threshold(probs: np.ndarray, y: np.ndarray, target_precision: float) -> float:
    """Lowest threshold (max coverage) whose accepted set still reaches the target precision."""
    order = np.argsort(-probs)
    best, correct = np.inf, 0
    for k, idx in enumerate(order, start=1):
        correct += y[idx]
        if k < len(order) and probs[order[k]] == probs[idx]:
            continue  # only cut between distinct scores
        if correct / k >= target_precision:
            best = float(probs[idx])
    return best


def evaluate(accept: np.ndarray, y: np.ndarray) -> dict:
    n_acc = int(accept.sum())
    return {
        "n": int(len(y)),
        "stp_rate": round(n_acc / len(y), 4),
        "accepted_precision": round(float(y[accept].mean()), 4) if n_acc else None,
        "errors_let_through": int((accept & (y == 0)).sum()),
    }


def rule_accept(X: np.ndarray, names: list[str], target: str) -> np.ndarray:
    """Baseline without ML: accept when both methods agree on every header field."""
    col = {n: i for i, n in enumerate(names)}
    accept = (X[:, col["n_agree_header"]] == 3) & (X[:, col["has_total"]] == 1)
    if target == "full":
        accept &= X[:, col["items_agreement_f1"]] == 1.0
    return accept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", default="llm_hybrid:claude-haiku-4.5:v2")
    parser.add_argument("--secondary", default="textract")
    parser.add_argument("--target", default="header", choices=["header", "full"])
    parser.add_argument("--precision", type=float, default=0.98, help="target precision of accepted docs")
    args = parser.parse_args()

    _, names, X_tr, y_tr = build_dataset(args.primary, args.secondary, "train", args.target)
    test_ids, _, X_te, y_te = build_dataset(args.primary, args.secondary, "test", args.target)

    print(f"Primary: {args.primary}   Secondary: {args.secondary}   Target: {args.target} correct")
    print(f"Train docs: {len(y_tr)} (accuracy {y_tr.mean():.3f})   Test docs: {len(y_te)} (accuracy {y_te.mean():.3f})")
    print(f"Target precision of auto-accepted documents: {args.precision:.2f}\n")

    report = {"args": vars(args), "features": names, "policies": {}}
    all_true_tr, all_true_te = np.ones(len(y_tr), bool), np.ones(len(y_te), bool)
    report["policies"]["accept_all"] = {"cv": evaluate(all_true_tr, y_tr), "test": evaluate(all_true_te, y_te)}
    report["policies"]["rule_all_fields_agree"] = {
        "cv": evaluate(rule_accept(X_tr, names, args.target), y_tr),
        "test": evaluate(rule_accept(X_te, names, args.target), y_te),
    }

    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=0
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    fitted = {}
    for name, model in models.items():
        oof = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        threshold = pick_threshold(oof, y_tr, args.precision)
        model.fit(X_tr, y_tr)
        p_te = model.predict_proba(X_te)[:, 1]
        fitted[name] = {"model": model, "threshold": threshold}
        report["policies"][name] = {
            "threshold": threshold,
            "cv": {**evaluate(oof >= threshold, y_tr), "auc": round(roc_auc_score(y_tr, oof), 4)},
            "test": {
                **evaluate(p_te >= threshold, y_te),
                "auc": round(roc_auc_score(y_te, p_te), 4) if len(set(y_te)) > 1 else None,
            },
            # Coverage curve on test for a README chart: (share auto-accepted, precision)
            "test_curve": [
                evaluate(p_te >= t, y_te) | {"threshold": round(float(t), 4)}
                for t in np.unique(np.round(p_te, 3))
            ],
        }

    header = "| Policy | CV STP | CV precision | Test STP | Test precision | Test errors let through | Test AUC |"
    print(header)
    print("|" + "---|" * (header.count("|") - 1))
    fmt = lambda v: "-" if v is None else f"{v:.3f}"
    for name, pol in report["policies"].items():
        cv_r, te_r = pol["cv"], pol["test"]
        print(
            f"| {name} | {fmt(cv_r['stp_rate'])} | {fmt(cv_r['accepted_precision'])} "
            f"| {fmt(te_r['stp_rate'])} | {fmt(te_r['accepted_precision'])} "
            f"| {te_r['errors_let_through']} | {fmt(te_r.get('auc'))} |"
        )

    lr = fitted["logistic_regression"]["model"][-1]
    top = sorted(zip(names, lr.coef_[0]), key=lambda kv: -abs(kv[1]))[:8]
    print("\nStrongest signals (logistic regression coefficients, standardized features):")
    for feat, coef in top:
        print(f"  {feat:<28} {coef:+.2f}")

    out_dir = RESULTS_DIR / "confidence" / f"{datetime.now():%Y%m%d-%H%M%S}_{args.target}_{args.precision}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    joblib.dump({"features": names, "models": fitted, "args": vars(args)}, out_dir / "models.joblib")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
