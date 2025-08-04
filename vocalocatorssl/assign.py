import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar


def numpy_softmax(x, axis=None, temperature=1.0):
    x = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x / temperature)
    return e_x / e_x.sum(axis=axis, keepdims=True)


def compute_ece(scores: np.ndarray) -> float:
    """Computes the calibration curve for the model on a dataset with known speaker identity.

    Args:
        scores (np.ndarray): Array of shape (batch, 1 + num_negative_samples) containing the scores for each sample.
            The first entry is expected to be the score of the positive sample, and the remaining entries are the scores
            of the negative samples for the same vocalization.

    Returns:
        float: The expected calibration error (ECE) for the model on the dataset.
    """
    num_bins = 20
    pos_scores = scores[:, 0]  # (B, )
    neg_scores = scores[:, 1:]  # (B, n_negative)

    num_negative = neg_scores.shape[1]
    pos_scores = pos_scores.repeat(num_negative, axis=0)  # (B * n_negative, )
    neg_scores = neg_scores.reshape(-1)  # (B * n_negative, )

    pos_neg_probs = np.stack([pos_scores, neg_scores], axis=1)  # (B * n_negative, 2)
    pos_neg_probs = numpy_softmax(pos_neg_probs, axis=1)

    Y_hat = pos_neg_probs.argmax(axis=1)
    P_hat = pos_neg_probs.max(axis=1)

    p_bins = np.linspace(0.5, 1.0, num_bins + 1, endpoint=True)
    bin_indices = np.digitize(P_hat, p_bins) - 1  # (B * n_negative, )
    bin_acc = np.zeros((num_bins,), dtype=float)
    bin_count = np.zeros((num_bins,), dtype=float)
    for j in range(num_bins):
        bin_mask = bin_indices == j
        bin_acc[j] = (Y_hat[bin_mask] == 0).mean() if np.any(bin_mask) else 0.0
        bin_count[j] = np.sum(bin_mask)

    bin_c = (p_bins[:-1] + p_bins[1:]) / 2

    ece = np.sum(
        np.abs(bin_acc - bin_c) * np.clip(bin_count, 1, None) / bin_count.sum()
    )
    return ece


def get_optimal_temperature_adjustment(test_scores: np.ndarray) -> float:
    def objective(temp):
        adjusted_scores = test_scores / temp
        ece = compute_ece(adjusted_scores)
        return ece

    result = minimize_scalar(objective, bounds=(0.5, 10))

    if not result.success:
        print(result.message)
        raise RuntimeError("Optimization failed to find a suitable temperature.")
    return result.x


def get_calibration_stats(test_results_path: Path) -> tuple[float, float, float]:
    """Extract calibration statistics from the model directory.
    Returns unadjusted ECE, adjusted ECE, and optimal temperature adjustment to obtain calibration

    Args:
        test_results_path (Path): Path to the test results file output by running inference with the --test flag.

    Returns:
        tuple[float, float, float]: Unadjusted ECE, adjusted ECE, and optimal temperature adjustment.
    """
    archive = np.load(test_results_path)
    score_keys = list(filter(lambda st: st.endswith("scores"), archive.files))
    orig_scores = np.concatenate(
        [archive[k] for k in score_keys], axis=0
    )  # (B, 1 + n_negative)

    unadjusted_ece = compute_ece(orig_scores)
    opt_temp = get_optimal_temperature_adjustment(orig_scores)
    adjusted_scores = orig_scores / opt_temp
    adjusted_ece = compute_ece(adjusted_scores)
    return unadjusted_ece, adjusted_ece, opt_temp


def get_assignments(
    prediction_path: Path, temp_adjustment: float = 1.0, threshold: float = 0.95
) -> dict[str, np.ndarray]:
    """Extract the assignment rate from the model directory."""
    # Locate the predictions file
    if not prediction_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {prediction_path}")

    data = np.load(prediction_path)
    fnames = list(map(bytes.decode, data["dataset_order"]))
    score_keys = [f"{fname}-scores" for fname in fnames]
    scores = [data[k] for k in score_keys]

    scores = [numpy_softmax(s, axis=1, temperature=temp_adjustment) for s in scores]
    top_scores = [s.max(axis=1) for s in scores]
    argmaxes = [s.argmax(axis=1) for s in scores]

    assignments = [
        np.where((ts >= threshold), amax, -1) for ts, amax in zip(top_scores, argmaxes)
    ]
    return {
        f"{fname}-assignments": assignment
        for fname, assignment in zip(fnames, assignments)
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", type=Path, help="Path to model --predict output")
    ap.add_argument(
        "--calibration-results",
        type=Path,
        help="Path to model --calibrate output, for computing calibrated assignments.",
    )
    ap.add_argument(
        "--conf-threshold",
        type=float,
        default=0.95,
        help="Necessary confidence in the animal's pose to consider the vocalization assignable.",
    )
    args = ap.parse_args()

    if args.conf_threshold <= 0.0 or args.conf_threshold >= 1.0:
        raise ValueError(
            "Confidence threshold must be in the range (0, 1). "
            f"Received: {args.conf_threshold}"
        )

    temp_adjustment = 1.0
    if args.calibration_results is not None:
        unadjusted_ece, adjusted_ece, opt_temp = get_calibration_stats(
            args.calibration_results
        )
        temp_adjustment = opt_temp
        print(f"ECE before calibration: {unadjusted_ece:.4f}")
        print(f"ECE after calibration: {adjusted_ece:.4f}")
        print(f"Optimal temperature adjustment: {temp_adjustment:.4f}")

    assignments = get_assignments(
        args.predictions, temp_adjustment=temp_adjustment, threshold=args.conf_threshold
    )
    num_assigned, num_total = 0, 0
    for fname, assignment in assignments.items():
        num_assigned += np.sum(assignment != -1)
        num_total += len(assignment)
    print(f"Num sessions: {len(assignments)}")
    print(f"Num vocalizations: {num_total}")
    print(
        f"Assignment rate ({args.conf_threshold:.0%} confidence): {num_assigned / num_total:.1%}"
    )

    data = np.load(args.predictions)
    # Append the assignments to the original data, removing old assignments if they exist
    data_files = list(filter(lambda st: not st.endswith("assignments"), data.files))
    np.savez(
        args.predictions,
        **{file: data[file] for file in data_files},
        **assignments,
        optimal_temperature_adjustment=np.array(temp_adjustment),
    )
