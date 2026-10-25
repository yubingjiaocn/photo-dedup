"""Two-frame, class-agnostic moving foreground candidates for manual review.

No detector, learned weights, semantic identity, completeness or keeper authority.
This deliberately narrow proposal path requires a mostly stationary background,
a unique large central changing color component in two segmentation granularities,
and reciprocal appearance/geometry agreement. It is not a primary-subject oracle.
"""
from __future__ import annotations

import hashlib

import cv2
import numpy as np

VERSION = "registered_color_component_pair_v1"
POLICY = {
    "analysis_max_edge": 1024, "kmeans_clusters": [8, 12], "rng_seed": 42,
    "kmeans_attempts": 3,
    "component_area_fraction": [.025, .30], "center_x": [.25, .75],
    "center_y": [.20, .80], "boundary_margin_fraction": .01,
    "changed_rgb_l2_threshold": 35, "changed_fraction_min": .50,
    "segmentation_box_iou_min": .75, "pair_box_iou_min": .40,
    "pair_mask_area_ratio_min": .60, "pair_center_distance_max": .15,
    "pair_lab_distance_max": 25, "pair_histogram_distance_max": .35,
    "registration_inliers_min": 40, "registration_inlier_ratio_min": .40,
    "registration_support_area_min": .30, "registration_scale": [.9, 1.1],
    "registration_median_error_max": 2.0, "registration_border_inliers_min": 20,
    "window_padding_fraction": .10, "strip_actual_short_edge_min": 256,
}


def _refuse(reason, **evidence):
    return {"status": "abstain", "reason": reason, "producer": VERSION,
            "review_only": True, "semantic_identity_authority": False,
            "keeper_authority": False, "bystander_face_crops": 0, **evidence}


def _iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    areas = [(x[2] - x[0]) * (x[3] - x[1]) for x in (a, b)]
    return intersection / max(1e-12, sum(areas) - intersection)


def _registration(images):
    gray = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in images]
    detector = cv2.SIFT_create(nfeatures=4000)
    a, da = detector.detectAndCompute(gray[0], None)
    b, db = detector.detectAndCompute(gray[1], None)
    if da is None or db is None or min(len(da), len(db)) < 40:
        return None, {"reason": "BACKGROUND_LOW_TEXTURE"}
    matches = [m for pair in cv2.BFMatcher().knnMatch(da, db, k=2)
               if len(pair) == 2 for m, n in [pair] if m.distance < .7 * n.distance]
    if len(matches) < 40:
        return None, {"reason": "BACKGROUND_MATCHES_LOW", "matches": len(matches)}
    p = np.float32([a[m.queryIdx].pt for m in matches])
    q = np.float32([b[m.trainIdx].pt for m in matches])
    matrix, inliers = cv2.estimateAffinePartial2D(p, q, method=cv2.RANSAC, ransacReprojThreshold=2)
    if matrix is None or inliers is None:
        return None, {"reason": "BACKGROUND_REGISTRATION_FAILED"}
    valid = inliers.ravel().astype(bool)
    h, w = gray[0].shape
    count = int(valid.sum())
    coverage = float(np.prod(np.ptp(p[valid], axis=0)) / (w * h))
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    error = float(np.median(np.linalg.norm(p[valid] @ matrix[:, :2].T + matrix[:, 2] - q[valid], axis=1)))
    border = int(np.count_nonzero((p[valid, 0] < w * .25) | (p[valid, 0] > w * .75)
                                  | (p[valid, 1] < h * .20) | (p[valid, 1] > h * .80)))
    info = {"matches": len(matches), "inliers": count, "inlier_ratio": count / len(matches),
            "support_area": coverage, "scale": scale, "median_error_px": error,
            "border_inliers": border, "source_to_target_affine": matrix.tolist()}
    if count < 40 or count / len(matches) < .4 or coverage < .3 or not .9 <= scale <= 1.1 or error > 2 or border < 20:
        return None, {"reason": "BACKGROUND_REGISTRATION_UNRELIABLE", **info}
    return matrix, info


def _components(image, peer, matrix, clusters):
    h, w = image.shape[:2]
    warped = cv2.warpAffine(peer, matrix, (w, h))
    valid_warp = cv2.warpAffine(np.ones((h, w), np.uint8), matrix, (w, h)) > 0
    delta = np.linalg.norm(cv2.GaussianBlur(image.astype("float32"), (9, 9), 0)
                           - cv2.GaussianBlur(warped.astype("float32"), (9, 9), 0), axis=2)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    cv2.setRNGSeed(POLICY["rng_seed"])
    _, labels, _ = cv2.kmeans(lab.reshape(-1, 3).astype("float32"), clusters, None,
                              (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, .2),
                              POLICY["kmeans_attempts"], cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(h, w)
    eligible, audit = [], []
    for cluster in range(clusters):
        mask = cv2.morphologyEx((labels == cluster).astype("uint8"), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, cc, stats, centers = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):
            x, y, bw, bh, area = map(int, stats[i])
            fraction = area / (w * h)
            if fraction < .025:
                continue
            cx, cy = centers[i] / [w, h]
            box = [x / w, y / h, (x + bw) / w, (y + bh) / h]
            selected = cc == i
            changed = float(np.mean((delta[selected] > 35) & valid_warp[selected]))
            reasons = []
            if fraction > .30:
                reasons.append("TOO_LARGE")
            if not .25 <= cx <= .75 or not .20 <= cy <= .80:
                reasons.append("OFF_CENTER")
            if min(box[0], box[1], 1 - box[2], 1 - box[3]) < .01:
                reasons.append("BOUNDARY_TOUCHING")
            if changed < .50:
                reasons.append("INSUFFICIENT_REGISTERED_CHANGE")
            record = {"cluster": cluster, "box_normalized": box, "area_fraction": fraction,
                      "centroid_normalized": [float(cx), float(cy)], "changed_fraction": changed,
                      "rejections": reasons}
            audit.append(record)
            if not reasons:
                hist = cv2.calcHist([lab], [1, 2], selected.astype("uint8"), [16, 16], [0, 256, 0, 256])
                hist /= max(float(hist.sum()), 1)
                eligible.append({**record, "mask": selected, "lab_mean": lab[selected].mean(axis=0), "hist": hist})
    return eligible, audit


def propose_pair(images):
    """Input is two oriented RGB uint8 arrays. Returned masks are analysis-scale."""
    if len(images) != 2 or any(im.dtype != np.uint8 or im.ndim != 3 or im.shape[2] != 3 for im in images):
        return _refuse("TWO_RGB_FRAMES_REQUIRED")
    if images[0].shape != images[1].shape:
        return _refuse("FRAME_GEOMETRY_CHANGED")
    if min(images[0].shape[:2]) < 256:
        return _refuse("NATIVE_IMAGE_TOO_SMALL")
    h, w = images[0].shape[:2]
    factor = min(1, 1024 / max(h, w))
    small = [cv2.resize(im, (round(w * factor), round(h * factor)), interpolation=cv2.INTER_AREA) for im in images]
    cv2.setRNGSeed(POLICY["rng_seed"])
    matrix, registration = _registration(small)
    if matrix is None:
        return _refuse(registration["reason"], registration=registration)
    selected, audit = [], []
    for index, image in enumerate(small):
        transform = cv2.invertAffineTransform(matrix) if index == 0 else matrix
        views = []
        for clusters in POLICY["kmeans_clusters"]:
            candidates, records = _components(image, small[1 - index], transform, clusters)
            audit.append({"frame_index": index, "clusters": clusters, "components": records})
            if len(candidates) != 1:
                return _refuse("NO_UNIQUE_CENTRAL_MOVING_COMPONENT", registration=registration, components=audit)
            views.append(candidates[0])
        overlap = _iou(views[0]["box_normalized"], views[1]["box_normalized"])
        if overlap < .75:
            return _refuse("SEGMENTATION_EXTENT_UNSTABLE", segmentation_box_iou=overlap, components=audit)
        # Union of the two segmentation extents; do not claim a whole-body mask.
        boxes = [v["box_normalized"] for v in views]
        candidate = {**views[0], "box_normalized": [min(b[0] for b in boxes), min(b[1] for b in boxes),
                                                       max(b[2] for b in boxes), max(b[3] for b in boxes)],
                     "segmentation_box_iou": overlap}
        selected.append(candidate)
    a, b = selected
    consistency = {
        "box_iou": _iou(a["box_normalized"], b["box_normalized"]),
        "mask_area_ratio": min(a["area_fraction"], b["area_fraction"]) / max(a["area_fraction"], b["area_fraction"]),
        "center_distance": float(np.linalg.norm(np.array(a["centroid_normalized"]) - b["centroid_normalized"])),
        "lab_mean_distance": float(np.linalg.norm(a["lab_mean"] - b["lab_mean"])),
        "histogram_distance": float(cv2.compareHist(a["hist"], b["hist"], cv2.HISTCMP_BHATTACHARYYA)),
        "reciprocal_unique": True,
    }
    if (consistency["box_iou"] < .4 or consistency["mask_area_ratio"] < .6 or consistency["center_distance"] > .15
            or consistency["lab_mean_distance"] > 25 or consistency["histogram_distance"] > .35):
        return _refuse("CROSS_FRAME_REGION_INCONSISTENT", consistency=consistency, components=audit)
    boxes = [item["box_normalized"] for item in selected]
    window = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
    padding = .10 * max(window[2] - window[0], window[3] - window[1])
    window = [window[0] - padding, window[1] - padding, window[2] + padding, window[3] + padding]
    if min(window) < 0 or max(window) > 1:
        return _refuse("PADDED_WINDOW_TRUNCATED", components=audit)
    records, masks = [], []
    for item in selected:
        mask = item["mask"].astype("uint8") * 255
        masks.append(mask)
        records.append({k: v for k, v in item.items() if k not in {"mask", "lab_mean", "hist"}})
        records[-1]["mask_pixels_sha256"] = hashlib.sha256(mask.tobytes()).hexdigest()
    return {"status": "candidate", "producer": VERSION, "policy": POLICY,
            "review_only": True, "semantic_identity_authority": False, "keeper_authority": False,
            "whole_body_completeness": "unknown", "bystander_face_crops": 0,
            "registration": registration, "components": audit, "regions": records,
            "consistency": consistency, "shared_window_normalized": window,
            "analysis_size": [small[0].shape[1], small[0].shape[0]], "masks": masks,
            "limitations": ["central moving color-coherent regions only", "background parallax or moving signage may cause abstention or false candidates",
                            "appearance/geometry consistency is not mascot identity or semantic action truth",
                            "human must confirm the action subject before production use"]}
