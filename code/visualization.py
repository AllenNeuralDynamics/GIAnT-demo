"""Visualizations for the GIAnT demo capsule.

1. Two-panel (Raw on top, Registered on bottom) comparison videos with
   time labels. Simplified version of comparison_video.ipynb / the
   paper's Movie S1 (Xie et al., "Glutamate Imaging Analysis
   Pipeline"): instead of the 2x2 Simulation/Scan x Raw/Registered
   grid, this makes one Raw/Registered video per scan, stacked
   vertically (matching Movie S1's Raw-on-top, Registered-on-bottom
   convention) rather than side by side -- these FOVs are strip-shaped
   (much wider than tall), so a side-by-side layout would make the
   video needlessly elongated.

2. A static figure of the SILo activity image with ROI contours
   overlaid. For the simulated dataset (which has ground truth), this
   shows matched/unmatched contours against the annotated ROIs, stacked
   vertically (matches on top, mismatches below) rather than side by
   side -- same strip-shaped-FOV reasoning as the video layout above.
   This is a fork of ``evaluate.py``'s registration/matching/plotting
   (``get_warp_matrix``, ``match``, ``plot_match``) rather than an
   import, since ``evaluate.py`` isn't shipped in this capsule. For
   datasets without ground truth (e.g. in vivo), it just overlays all
   detected ROI contours with no matching.

3. A per-ROI fluorescence trace figure, zoomed into a short,
   representative time window -- like the paper's Figure 4b/c --
   rather than the full recording, which is what ``evaluate.py``'s own
   ``plot_traces`` shows (unreadable at this frame count). Works for
   any dataset, no ground truth needed.

Expects the classic capsule layout, with this script living in code/
alongside data/ and results/:
    ../data/<dataset>/<raw file>.h5|.tif
    ../data/<dataset>/<name>_groundtruth.h5                           (simulated only)
    ../results/<dataset>/motion_correction/<name>_REGISTERED_RAW.h5
    ../results/<dataset>/motion_correction/<name>_ALIGNMENTDATA.h5    (optional, for frame_time)
    ../results/<dataset>/source_extraction/experiment_summary.h5

Outputs are written next to the files they're derived from.
"""

import os

import cv2
import h5py
import imageio_ffmpeg
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from aind_ophys_utils.array_utils import downsample_array
from matplotlib.colors import PowerNorm
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from scipy import sparse as sp
from scipy.optimize import linear_sum_assignment
from skimage.registration import phase_cross_correlation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")

DEFAULT_FRAME_TIME = 0.0023  # seconds/frame, used if ALIGNMENTDATA.h5 is missing
ACT_IM_QUANTILES = (1, 99.8)  # percentile clip for the activity-image display range


def _video_paths(
    dataset: str, name: str, raw_filename: str, groundtruth_filename: str = None
) -> dict:
    """Build the path dict consumed throughout this script: ``name``, ``raw``,
    ``registered``, ``alignment``, ``source_extraction``, and ``groundtruth``
    (``None`` if ``groundtruth_filename`` isn't given, e.g. for in vivo).
    """
    motion_dir = os.path.join(RESULTS_DIR, dataset, "motion_correction")
    return {
        "name": name,
        "raw": os.path.join(DATA_DIR, dataset, raw_filename),
        "registered": os.path.join(motion_dir, f"{name}_REGISTERED_RAW.h5"),
        "alignment": os.path.join(motion_dir, f"{name}_ALIGNMENTDATA.h5"),
        "source_extraction": os.path.join(
            RESULTS_DIR, dataset, "source_extraction", "experiment_summary.h5"
        ),
        "groundtruth": (
            None
            if groundtruth_filename is None
            else os.path.join(DATA_DIR, dataset, groundtruth_filename)
        ),
    }


VIDEOS = [
    _video_paths(
        "simulated",
        "SIMULATION_scan_00005-5_Trial1_demo",
        "SIMULATION_scan_00005-5_Trial1_demo.h5",
        groundtruth_filename="SIMULATION_scan_00005-5_Trial1_demo_groundtruth.h5",
    ),
    _video_paths("in_vivo", "scan_00001_demo", "scan_00001_demo.tif"),
]


def load_movie(path: str) -> np.ndarray:
    """Load a (T, H, W) movie from a .tif/.tiff or .h5 (dataset "data") file."""
    if path.lower().endswith((".tif", ".tiff")):
        return tifffile.imread(path)
    with h5py.File(path, "r") as f:
        return f["data"][:]


def frame_time_from_alignment(path: str, default: float = DEFAULT_FRAME_TIME) -> float:
    if os.path.exists(path):
        with h5py.File(path, "r") as f:
            return float(np.asarray(f["frametime"]).ravel()[0])
    return default


def center_crop(movie: np.ndarray, dims: tuple) -> np.ndarray:
    h, w = movie.shape[1:]
    dh, dw = dims
    r0 = max((h - dh) // 2, 0)
    c0 = max((w - dw) // 2, 0)
    return movie[:, r0 : r0 + dh, c0 : c0 + dw]


def _load_source_extraction(path: str) -> tuple:
    """Load (footprints, activity image, mean image, SNR) from an
    ``experiment_summary.h5``. Activity image is in its original units --
    display code applies a sqrt-law color compression (see ``_activity_norm``).
    """
    with h5py.File(path, "r") as f:
        footprints = f["Path1/sources/spatial/profiles"][:, :, 0].transpose(2, 0, 1).astype("f4")
        act_im = np.squeeze(f["Path1/visualizations/act_im"][:]).astype("f4")
        mean_im = np.squeeze(f["Path1/visualizations/mean_im"][:]).astype("f4")
        snr = np.squeeze(f["Path1/sources/temporal/SNR"][:])
    return footprints, act_im, mean_im, snr


def _activity_norm(im: np.ndarray, q: tuple) -> "PowerNorm":
    """Sqrt-law color normalization for an activity image (long-tailed
    dynamic range -- a linear scale would hide faint ROIs next to bright
    ones), percentile-clipped to ``q``. Applied at display time so the data
    (and a colorbar built from it) stay in the activity image's original units.
    """
    lp, hp = np.nanpercentile(im, q)
    return PowerNorm(gamma=1, vmin=max(lp, 0), vmax=hp)


def _snr_rank(snr: np.ndarray, subset: set = None) -> np.ndarray:
    """1-based rank by descending SNR (1 = highest), indexed by each ROI's
    original index. If ``subset`` is given, only those are ranked
    (contiguously); everything else gets 0 ("unranked") -- used to number
    only true-positive ROIs, keeping the numbering gap-free.
    """
    idx = np.arange(len(snr)) if subset is None else np.array(sorted(subset))
    order = idx[np.argsort(snr[idx])[::-1]]
    rank = np.zeros(len(snr), dtype=int)
    rank[order] = np.arange(1, len(order) + 1)
    return rank


def _pearsonr(x: np.ndarray, y: np.ndarray, min_valid: int = 1) -> float:
    """Pearson correlation over entries where both ``x`` and ``y`` are finite,
    or NaN if fewer than ``min_valid`` such entries exist. Ported from the
    ``pearsonr`` closure in ``evaluate.py``'s ``evaluate()`` (faster than
    ``scipy.stats.pearsonr``); also used for the image-alignment correlation
    in ``findTransformXcorr``/``normalized_correlation``.
    """
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < min_valid:
        return np.nan
    x, y = x[valid], y[valid]
    xm, ym = x - x.mean(), y - y.mean()
    return np.dot(xm, ym) / np.sqrt(np.dot(xm, xm) * np.dot(ym, ym))


def _valid_bbox(im: np.ndarray) -> tuple:
    """Row/col slices bounding the region of ``im`` that isn't entirely NaN
    (crops away the registration border-trim margin)."""
    valid_rows = ~np.all(np.isnan(im), axis=1)
    valid_cols = ~np.all(np.isnan(im), axis=0)
    r0, r1 = np.flatnonzero(valid_rows)[[0, -1]]
    c0, c1 = np.flatnonzero(valid_cols)[[0, -1]]
    return slice(r0, r1 + 1), slice(c0, c1 + 1)


def _footprint_centroid(footprint: np.ndarray) -> tuple:
    """Weighted centroid ``(row, col)`` of a spatial footprint, or ``None``
    if it has no positive weight (e.g. all-zero/NaN after alignment)."""
    weights = np.clip(np.nan_to_num(footprint), 0, None)
    total = weights.sum()
    if total <= 0:
        return None
    ys, xs = np.indices(footprint.shape)
    return (weights * ys).sum() / total, (weights * xs).sum() / total


def _label_roi(ax, footprint: np.ndarray, label: str) -> None:
    """Draw ``label`` at ``footprint``'s centroid on ``ax``, matching the
    SNR-rank numbering used for that ROI in ``make_trace_figure``.
    """
    centroid = _footprint_centroid(footprint)
    if centroid is None:
        return
    cy, cx = centroid
    ax.text(
        cx,
        cy,
        label,
        color="C6",
        fontsize=8,
        ha="center",
        va="center",
        path_effects=[pe.withStroke(linewidth=1.5, foreground="k")],
    )


def _draw_contours(rois: np.ndarray, color: str) -> None:
    """Overlay a normalized 0.2-level contour for each ROI/footprint in ``rois``."""
    for roi in rois:
        plt.contour(
            roi / (roi.max() + np.finfo(np.float32).eps), levels=[0.2], colors=color, linewidths=2
        )


def findTransformXcorr(
    templateImage: np.ndarray, inputImage: np.ndarray, max_shift: int = 30
) -> tuple:
    """Rough translation between two images by normalized cross-correlation
    over integer pixel shifts. Returns (correlation, 2x3 affine warp matrix).
    """
    shifts = np.arange(-max_shift, max_shift + 1)
    C = np.full((len(shifts), len(shifts)), np.nan)
    for drix in range(len(shifts)):
        for dcix in range(len(shifts)):
            T = cv2.warpAffine(
                templateImage,
                np.float32([[1, 0, shifts[dcix]], [0, 1, shifts[drix]]]),
                templateImage.shape[::-1],
                cv2.BORDER_CONSTANT,
                borderValue=np.nan,
            )
            C[drix, dcix] = _pearsonr(T, inputImage, min_valid=10)
    maxval = np.nanmax(C)
    rr, cc = np.unravel_index(np.nanargmax(C), C.shape)
    return float(maxval), np.array([[1, 0, shifts[cc]], [0, 1, shifts[rr]]], dtype="f4")


def normalized_correlation(im1: np.ndarray, im2: np.ndarray, warp_matrix: np.ndarray) -> float:
    """Pearson correlation between im1 and im2 warped onto it, or NaN if no
    valid overlap."""
    im2_shifted = cv2.warpAffine(
        im2,
        warp_matrix,
        im1.shape[::-1],
        cv2.BORDER_CONSTANT,
        borderValue=np.nan,
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
    )
    return _pearsonr(im1, im2_shifted)


def get_warp_matrix(results: dict, GTmeanIM: np.ndarray, max_shift: int = 30) -> np.ndarray:
    """Best affine warp aligning ``results["meanIM"]`` onto ``GTmeanIM``,
    picked among cross-correlation, ECC, and phase-correlation candidates
    by whichever gives the highest normalized correlation.
    """
    print("Computing warp matrix")
    tmp = results["meanIM"].astype("f4")
    dims = tuple(max(*d) for d in zip(GTmeanIM.shape, tmp.shape))
    im1, im2 = np.nan * np.zeros((2,) + dims, dtype="f4")
    im1[: GTmeanIM.shape[0], : GTmeanIM.shape[1]] = GTmeanIM[:]
    im2[: tmp.shape[0], : tmp.shape[1]] = tmp

    warp_matrices = [findTransformXcorr(im1, im2, max_shift)[1], None]
    ecc_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-3)
    try:
        warp_matrices[1] = cv2.findTransformECC(
            np.nan_to_num(im1),
            np.nan_to_num(im2),
            warp_matrices[0].copy(),
            cv2.MOTION_TRANSLATION,
            ecc_criteria,
            None,
            5,
        )[1]
    except Exception:
        pass
    shift = -phase_cross_correlation(np.nan_to_num(im1), np.nan_to_num(im2), upsample_factor=100)[0]
    warp_matrices.append(np.array([[1, 0, shift[1]], [0, 1, shift[0]]]))

    corrs = [-1 if wm is None else normalized_correlation(im1, im2, wm) for wm in warp_matrices]
    return warp_matrices[np.argmax(corrs)]


def multicolor_title(
    ax, list_of_strings: list, list_of_colors: list, anchorpad: float = 0, **kw
) -> None:
    """Multicolored title anchored above ``ax``."""
    boxes = [
        TextArea(text, textprops=dict(color=color, ha="left", va="bottom", **kw))
        for text, color in zip(list_of_strings, list_of_colors)
    ]
    xbox = HPacker(children=boxes, align="center", pad=0, sep=5)
    anchored_xbox = AnchoredOffsetbox(
        loc="upper center",
        child=xbox,
        pad=anchorpad,
        frameon=False,
        bbox_to_anchor=(0.5, 1.07),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    ax.add_artist(anchored_xbox)


def match(gtROIs: np.ndarray, eROIs: np.ndarray, thresh_s: float = 0.5) -> tuple:
    """Match ground-truth to estimated ROIs by spatial (cosine) overlap via
    linear-sum-assignment. Returns (matched_ROIs1, matched_ROIs2,
    non_matched1, non_matched2, performance, overlap).
    """
    N, d = gtROIs.shape[0], np.prod(eROIs.shape[1:])
    A = sp.csc_matrix(gtROIs.reshape((N, d)), dtype="f4")
    A = sp.diags(1 / sp.linalg.norm(A, 2, 1)).dot(A)
    eA = sp.csc_matrix(eROIs.reshape((eROIs.shape[0], d)), dtype="f4")
    eA = sp.diags(1 / (sp.linalg.norm(eA, 2, 1) + np.finfo(np.float32).eps)).dot(eA)
    olap = A.dot(eA.T).toarray()
    # below is inspired from stardist
    n_matched = min(A.shape[0], eA.shape[0])
    costs = -(olap >= thresh_s).astype(float) - olap / (2 * n_matched)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = olap[true_ind, pred_ind] >= thresh_s
    matched_ROIs1 = true_ind[match_ok]
    matched_ROIs2 = pred_ind[match_ok]
    non_matched1 = np.setdiff1d(range(N), matched_ROIs1).astype(int)
    non_matched2 = np.setdiff1d(range(eROIs.shape[0]), matched_ROIs2).astype(int)
    TP, FN, FP = len(matched_ROIs1), len(non_matched1), len(non_matched2)
    performance = {
        "recall": np.nan if TP + FN == 0 else TP / (TP + FN),
        "precision": np.nan if TP + FP == 0 else TP / (TP + FP),
        "f1_score": np.nan if (den := 2 * TP + FP + FN) == 0 else 2 * TP / den,
    }
    return (
        matched_ROIs1,
        matched_ROIs2,
        non_matched1,
        non_matched2,
        performance,
        olap[matched_ROIs1, matched_ROIs2],
    )


def plot_match(
    gtROIs: np.ndarray,
    eROIs: np.ndarray,
    matched_ROIs1: np.ndarray,
    matched_ROIs2: np.ndarray,
    non_matched1: np.ndarray,
    non_matched2: np.ndarray,
    im: np.ndarray,
    q: tuple = (5, 97),
) -> None:
    """Activity image with matched (top panel) / mismatched (bottom panel)
    ROI contours overlaid. ``evaluate.py``'s version places these side by
    side; stacked here since the strip-shaped FOV makes side-by-side
    needlessly elongated (see ``make_two_panel_video``).
    """
    cTPgt, cTPseg, cFN, cFP = "C2", "C6", "C1", "C3"
    norm = _activity_norm(im, q)

    _, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(8, 7))

    plt.sca(ax_top)
    im_top = plt.imshow(im, norm=norm, cmap="gray")
    _draw_contours(gtROIs[matched_ROIs1], cTPgt)
    _draw_contours(eROIs[matched_ROIs2], cTPseg)
    multicolor_title(
        ax_top,
        list_of_strings=["Matches:  ", "annotated", "&", "segmented", " true positives"],
        list_of_colors=["k", cTPgt, "k", cTPseg, "k"],
        fontsize=16,
    )
    plt.axis("off")

    plt.sca(ax_bottom)
    plt.imshow(im, norm=norm, cmap="gray")
    _draw_contours(gtROIs[non_matched1], cFN)
    _draw_contours(eROIs[non_matched2], cFP)
    multicolor_title(
        ax_bottom,
        list_of_strings=["Mismatches:  ", "false positives", "&", "false negatives"],
        list_of_colors=["k", cFP, "k", cFN],
        fontsize=16,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.gcf().colorbar(
        im_top, ax=[ax_top, ax_bottom], label="activity (A.U.)", fraction=0.05, shrink=0.8
    )


def _fit_trace_scale(extracted: np.ndarray, true: np.ndarray) -> tuple:
    """Affine (offset, slope) to overlay ``extracted`` onto ``true``'s scale.
    Ported from ``evaluate.py``'s ``plot_traces``: intercept from a
    least-squares fit, slope from the ratio of the two traces' ranges.
    """
    valid = np.isfinite(extracted) & np.isfinite(true)
    e, tr = extracted[valid], true[valid]
    em, tm = e.mean(), tr.mean()
    m = np.sum((e - em) * (tr - tm)) / np.sum((e - em) ** 2)
    c = tm - m * em
    m = (tr.max() - tr.min()) / (e.max() - e.min() + np.finfo(np.float32).eps)
    return c, m


def make_activity_roi_figure(
    raw_path: str,
    source_extraction_path: str,
    groundtruth_path: str,
    output_path: str,
    match_thresh: float = 0.5,
) -> dict:
    """Plot the SILo activity image with matched/unmatched ROI contours,
    comparing extracted footprints against simulation ground truth (like
    the ``actIM`` step of ``evaluate.py``'s ``evaluate()``).

    Ground truth is in the raw (unregistered) frame, extracted
    footprints/activity image in the registered (padded) frame, so the
    latter are aligned via a warp matrix estimated between the two mean
    images -- using the raw movie's own temporal mean as the reference
    (rather than ``evaluate.py``'s synthetic noise-free one) to keep this
    script self-contained.

    Returns ``{extracted ROI index: matched ground-truth ROI index}`` for
    reuse by ``make_trace_figure``.
    """
    with h5py.File(groundtruth_path, "r") as f:
        gt_rois_full = f["GT/ROIs"][:]  # (Z, H, W, n_rois), raw/unregistered frame
    gt_rois = gt_rois_full[gt_rois_full.shape[0] // 2].transpose(2, 0, 1)
    dims = gt_rois.shape[1:]

    footprints, act_im, mean_im, snr = _load_source_extraction(source_extraction_path)

    gt_mean_im = np.mean(load_movie(raw_path), axis=0).astype("f4")
    warp_matrix = get_warp_matrix({"meanIM": mean_im}, gt_mean_im)

    def align(im):
        return cv2.warpAffine(
            im, warp_matrix, dims[::-1], flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )[: dims[0], : dims[1]]

    footprints_aligned = np.nan_to_num([align(f) for f in footprints])
    act_im_aligned = align(act_im)

    TPgt, TPseg, FN, FP, performance, _ = match(gt_rois, footprints_aligned, match_thresh)

    # For display only (not the matching above): crop to the region of
    # act_im_aligned that's actually valid (see _valid_bbox).
    rows, cols = _valid_bbox(act_im_aligned)
    gt_rois_disp = gt_rois[:, rows, cols]
    footprints_disp = footprints_aligned[:, rows, cols]
    act_im_disp = act_im_aligned[rows, cols]

    plot_match(
        gt_rois_disp, footprints_disp, TPgt, TPseg, FN, FP, im=act_im_disp, q=ACT_IM_QUANTILES
    )

    # Number true-positive ROIs (contiguously, 1..#TP) to match
    # make_trace_figure's labels, which only shows TP ROIs -- false
    # positives aren't labeled here either, so numbering stays consistent
    # between the two figures.
    tpseg = set(TPseg.tolist())
    rank = _snr_rank(snr, subset=tpseg)
    ax_match = plt.gcf().axes[0]
    for i, footprint in enumerate(footprints_disp):
        if i in tpseg:
            _label_roi(ax_match, footprint, str(rank[i]))

    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(
        f"Wrote {output_path} "
        f"(precision={performance['precision']:.3f}, recall={performance['recall']:.3f}, "
        f"f1={performance['f1_score']:.3f})"
    )
    return dict(zip(TPseg.tolist(), TPgt.tolist()))


def make_activity_figure(
    source_extraction_path: str,
    output_path: str,
    q: tuple = ACT_IM_QUANTILES,
) -> None:
    """Plot the SILo activity image with detected ROI contours overlaid.

    For datasets with no ground truth (e.g. in vivo), where matching
    against annotated ROIs (see ``make_activity_roi_figure``) isn't
    possible -- this just shows what SILo found.
    """
    footprints, act_im, _, snr = _load_source_extraction(source_extraction_path)
    rank = _snr_rank(snr)

    rows, cols = _valid_bbox(act_im)
    act_im = act_im[rows, cols]
    footprints = footprints[:, rows, cols]

    norm = _activity_norm(act_im, q)
    plt.figure(figsize=(8, 3.5))
    ax = plt.gca()
    im_artist = plt.imshow(act_im, norm=norm, cmap="gray")
    _draw_contours(footprints, "C6")
    for i, fp in enumerate(footprints):
        _label_roi(ax, fp, str(rank[i]))  # matches make_trace_figure's SNR-rank labels
    plt.colorbar(im_artist, ax=ax, label="activity (A.U.)", fraction=0.03, shrink=0.7)
    plt.title(f"SILo activity image with {len(footprints)} detected ROIs")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"Wrote {output_path} ({len(footprints)} ROIs)")


def make_trace_figure(
    source_extraction_path: str,
    output_path: str,
    frame_time: float,
    window_s: float = 15.0,
    start_s: float = 10.0,
    trace_key: str = "dF_denoised",
    groundtruth_path: str = None,
    matched_gt: dict = None,
) -> None:
    """Plot each detected ROI's fluorescence trace, zoomed into a short,
    fixed ``start_s``-``start_s + window_s`` window (like the paper's
    Figure 4b/c) rather than the full ~78 s recording, which ``evaluate.py``'s
    own ``plot_traces`` shows and which is unreadable at this frame count.
    Traces are sorted by SNR (highest first).

    When ``groundtruth_path``/``matched_gt`` (from ``make_activity_roi_figure``)
    are given, only matched (true-positive) ROIs are shown, each with its
    true trace overlaid (scaled onto the extracted trace's range). Both
    ``None`` for datasets without ground truth, which shows all ROIs.
    """
    with h5py.File(source_extraction_path, "r") as f:
        traces = f[f"Path1/sources/temporal/{trace_key}"][:, 0, :].T  # (n_rois, T)
        snr = np.squeeze(f["Path1/sources/temporal/SNR"][:])

    gt_activity = None
    if groundtruth_path is not None:
        with h5py.File(groundtruth_path, "r") as f:
            gt_activity = f["GT/activity"][:].astype("f4")  # (n_gt_rois, T)

    # Rank (and, when ground truth is given, filter to) true positives only,
    # matching make_activity_roi_figure's numbering exactly: both compute
    # the same SNR-based rank restricted to the same TP index set.
    rank = _snr_rank(snr, subset=None if matched_gt is None else set(matched_gt))
    order = np.argsort(snr)[::-1]
    if matched_gt is not None:
        order = np.array([i for i in order if i in matched_gt])
    traces = traces[order]

    window_frames = int(round(window_s / frame_time))
    start = int(round(start_s / frame_time))
    window = traces[:, start : start + window_frames]
    t = np.arange(window_frames) * frame_time

    n = len(traces)
    _, ax = plt.subplots(n, 1, sharex=True, figsize=(10, 0.5 + 0.4 * n))
    has_legend = False
    for i, tr in enumerate(window):
        plt.sca(ax[i])
        roi_idx = order[i]
        gt_idx = None if matched_gt is None else matched_gt.get(roi_idx)
        if gt_activity is not None and gt_idx is not None:
            gC = gt_activity[gt_idx]
            c, m = _fit_trace_scale(traces[i], gC)
            r = _pearsonr(traces[i], gC)  # over the full trace, not just this window
            plt.plot(t, gC[start : start + window_frames], lw=1.5, c="C2", label="True trace")
            plt.plot(t, c + m * tr, lw=0.75, c="C6", label="Extracted trace")
            plt.text(
                0.99,
                0.85,
                f"r = {r:.3f}",
                transform=ax[i].transAxes,
                ha="right",
                va="top",
                c="C2",
                fontsize=7,
                bbox=dict(facecolor="w", edgecolor="none", pad=1),
            )
            if not has_legend:
                plt.legend(ncol=2, loc=(0.6, 1.05), frameon=False, fontsize=8)
                has_legend = True
        else:
            plt.plot(t, tr, lw=0.75, c="C6")
        plt.yticks([])
        plt.ylabel(f"#{rank[roi_idx]}", rotation=0, labelpad=20, va="center", fontsize=8, c="C6")
        for side in ("top", "right", "left"):
            ax[i].spines[side].set_visible(False)
    plt.xlim(0, window_s)
    plt.xlabel("Time [s]")
    plt.suptitle(
        f"{trace_key.removeprefix('dF_')} traces, {window_s:.0f} s window at "
        f"t={start * frame_time:.1f} s (sorted by SNR)",
        y=1.0,
    )
    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(hspace=0.2)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"Wrote {output_path} ({n} ROIs, window {window_s:.0f}s @ t={start * frame_time:.1f}s)")


def make_two_panel_video(
    raw_path: str,
    registered_path: str,
    output_path: str,
    frame_time: float = DEFAULT_FRAME_TIME,
    downscale: int = 20,
    lower_quantile: float = 0.02,
    upper_quantile: float = 0.9975,
    n_jobs: int = None,
    bitrate: str = "0",
    crf: int = 32,
    cpu_used: int = 4,
) -> None:
    """Encode a Raw (top) / Registered (bottom) video with a burned-in time label.

    Parameters mirror the ``video()`` helper in comparison_video.ipynb.
    """
    print(f"Loading raw movie: {raw_path}")
    raw = load_movie(raw_path)
    print(f"Loading registered movie: {registered_path}")
    registered = load_movie(registered_path)

    # Align frame counts and spatial dims in case raw/registered differ slightly.
    n = min(len(raw), len(registered))
    raw, registered = raw[:n], registered[:n]
    dims = (
        min(raw.shape[1], registered.shape[1]),
        min(raw.shape[2], registered.shape[2]),
    )
    raw = center_crop(raw, dims)
    registered = center_crop(registered, dims)

    raw_ds, registered_ds = [
        downsample_array(m, factors=downscale, n_jobs=n_jobs) for m in (raw, registered)
    ]
    panel_h, panel_w = raw_ds.shape[1:]

    fs = 1 / frame_time / downscale  # real-time playback speed

    combined = np.concatenate((raw_ds, registered_ds), axis=1)
    minmov, maxmov = np.nanquantile(
        combined[:: max(1, len(combined) // 100)], (lower_quantile, upper_quantile)
    )

    def scale(m):
        return np.nan_to_num(np.clip(255 * (m - minmov) / (maxmov - minmov), 0, 255)).astype(
            np.uint8
        )

    raw_ds = scale(raw_ds)
    registered_ds = scale(registered_ds)

    # scale movie for display (magnify based on a single panel's height)
    magnify = max(600 // panel_h, 1)
    h, w = panel_h * magnify, panel_w * magnify  # single-panel size, post magnify

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = min(h / 600, w / 190)
    thickness = max(round(fontscale * 2), 1)
    label_textheight = cv2.getTextSize("Registered", font, fontscale, thickness)[0][1]
    strip_h = label_textheight + 12  # blank strip reserved for each panel label

    data_h = 2 * h + 2 * strip_h
    canvas_h = int(np.ceil(data_h / 16)) * 16
    canvas_w = int(np.ceil(w / 16)) * 16
    canvas = np.zeros((canvas_h, canvas_w), np.uint8)
    top_pad = canvas_h - data_h  # rounding slack from the /16 ceiling, usually 0

    raw_y0 = top_pad + strip_h
    raw_y1 = raw_y0 + h
    reg_y0 = raw_y1 + strip_h
    reg_y1 = reg_y0 + h  # == canvas_h

    # panel labels live in their own blank strip (never touched by the frame
    # paste below), so they're drawn once and don't overlap image data.
    for label, y0 in (("Raw", top_pad), ("Registered", raw_y1)):
        cv2.putText(
            canvas,
            label,
            (8, y0 + label_textheight + 2),
            font,
            fontscale,
            (255,),
            thickness,
            cv2.LINE_4,
        )

    time_textsize = cv2.getTextSize("Time  000.0s", font, fontscale, thickness)[0]

    writer = imageio_ffmpeg.write_frames(
        output_path,
        # ffmpeg expects video shape in terms of: (width, height)
        (canvas_w, canvas_h),
        pix_fmt_in="gray8",
        pix_fmt_out="yuv420p",
        codec="libvpx-vp9",
        fps=fs,
        bitrate=bitrate,
        output_params=[
            "-crf",
            str(crf),
            "-row-mt",
            "1",
            "-cpu-used",
            str(cpu_used),
        ],
    )
    writer.send(None)  # Seed ffmpeg-imageio writer generator
    for t, (raw_frame, reg_frame) in enumerate(zip(raw_ds, registered_ds)):
        if magnify > 1:
            raw_frame = cv2.resize(raw_frame, (0, 0), fx=magnify, fy=magnify)
            reg_frame = cv2.resize(reg_frame, (0, 0), fx=magnify, fy=magnify)
        canvas[raw_y0:raw_y1, -w:] = raw_frame
        canvas[reg_y0:reg_y1, -w:] = reg_frame
        # Time is burned directly onto the data (bottom-right of the
        # Registered panel), matching the notebook/paper's own convention.
        text = f"Time {t * frame_time * downscale:6.1f}s"
        cv2.putText(
            canvas,
            text,
            (canvas_w - time_textsize[0] - 8, canvas_h - 8),
            font,
            fontscale,
            (255,),
            thickness,
            cv2.LINE_4,
        )
        writer.send(canvas)
    writer.close()
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    for v in VIDEOS:
        output_path = os.path.join(os.path.dirname(v["registered"]), "raw_vs_registered.webm")
        frame_time = frame_time_from_alignment(v["alignment"])
        make_two_panel_video(v["raw"], v["registered"], output_path, frame_time=frame_time)

        source_extraction_dir = os.path.dirname(v["source_extraction"])
        fig_path = os.path.join(source_extraction_dir, "activity_with_rois.pdf")
        matched_gt = None
        if v["groundtruth"] is not None:
            matched_gt = make_activity_roi_figure(
                v["raw"], v["source_extraction"], v["groundtruth"], fig_path
            )
        else:
            make_activity_figure(v["source_extraction"], fig_path)

        traces_path = os.path.join(source_extraction_dir, "traces.pdf")
        make_trace_figure(
            v["source_extraction"],
            traces_path,
            frame_time,
            groundtruth_path=v["groundtruth"],
            matched_gt=matched_gt,
        )
