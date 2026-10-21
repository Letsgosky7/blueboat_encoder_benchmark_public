"""Construct matched seabed anchors and geocoded positive/negative sonar patches.

Input: prepared survey geometry, sonar samples, colocation candidates and anchors.
Output: geocoded log1p mosaics, masks, matching tables and patch-bank arrays.
Usage: imported by DatasetBuilder; no standalone CLI."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class AnchorConfig:
    """Sampling intervals and geometric matching tolerances for anchor candidates."""
    row_stride: int = 20
    ground_range_step_m: float = 1.0
    min_ground_range_m: float = 2.0
    far_edge_margin_m: float = 1.0
    max_candidate_distance_m: float = 10.0
    distance_thresholds_m: tuple = (0.5, 1.0, 2.0, 5.0, 10.0)


def angle_difference_deg(angle_a, angle_b):
    """Return the smaller of clockwise/counterclockwise angular differences."""
    a, b = np.asarray(angle_a, dtype=np.float64), np.asarray(angle_b, dtype=np.float64)
    difference = np.abs(a - b) % 360.0
    return np.minimum(difference, 360.0 - difference)


def nearest_axis_indices(axis, values):
    """Find the closest slant-range sample index for each requested distance."""
    axis, values = np.asarray(axis, dtype=np.float64), np.asarray(values, dtype=np.float64)
    # searchsorted finds insertion positions on the sorted axis; check both neighbors.
    right = np.clip(np.searchsorted(axis, values, side="left"), 0, len(axis) - 1)
    left = np.clip(right - 1, 0, len(axis) - 1)
    return np.where(np.abs(values - axis[left]) <= np.abs(axis[right] - values), left, right).astype(np.int32)


def geocode_ground_range(sensor_east, sensor_north, heading_deg, ground_range, side):
    """Convert a heading, side and horizontal range to UTM east/north."""
    theta = np.deg2rad(np.asarray(heading_deg, dtype=np.float64))
    ground_range = np.asarray(ground_range, dtype=np.float64)
    if side == "starboard":
        direction_east, direction_north = np.cos(theta), -np.sin(theta)
    elif side == "port":
        direction_east, direction_north = -np.cos(theta), np.sin(theta)
    else:
        raise ValueError("side must be 'port' or 'starboard'.")
    return sensor_east + ground_range * direction_east, sensor_north + ground_range * direction_north


def look_azimuth_deg(heading_deg, side):
    """Convert vessel heading plus sonar side into seabed look direction."""
    heading_deg = np.asarray(heading_deg, dtype=np.float64)
    if side == "starboard":
        return (heading_deg + 90.0) % 360.0
    if side == "port":
        return (heading_deg - 90.0) % 360.0
    raise ValueError("side must be 'port' or 'starboard'.")


def build_anchors(geometry, survey_name, config=AnchorConfig()):
    """Sample confident seabed points along approved survey legs."""
    row_index = geometry["row_index"].astype(np.int32)
    ping_index = geometry["ping_index"].astype(np.int64)
    east, north = geometry["east"].astype(np.float64), geometry["north"].astype(np.float64)
    heading = geometry["heading_deg"].astype(np.float64)
    leg_number = geometry["leg_number"].astype(np.int32)
    in_analysis_leg = geometry["in_analysis_leg"].astype(bool)
    h_port, h_stbd = geometry["first_bottom_port"].astype(np.float64), geometry["first_bottom_stbd"].astype(np.float64)
    score_port, score_stbd = geometry["first_bottom_score_port"].astype(np.float64), geometry["first_bottom_score_stbd"].astype(np.float64)
    high_port, high_stbd = geometry["high_confidence_port"].astype(bool), geometry["high_confidence_stbd"].astype(bool)
    axis_port, axis_stbd = geometry["canonical_slant_axis_port"].astype(np.float64), geometry["canonical_slant_axis_stbd"].astype(np.float64)
    gmax_port, gmax_stbd = geometry["max_ground_range_port"].astype(np.float64), geometry["max_ground_range_stbd"].astype(np.float64)
    utm_epsg = int(np.asarray(geometry["utm_epsg"]).item())

    columns = {name: [] for name in (
        "survey", "row_index", "ping_index", "leg_number", "side", "canonical_sample_index", "raw_sample_index",
        "slant_range_m", "ground_range_m", "first_bottom_m", "first_bottom_score", "high_confidence",
        "sensor_east", "sensor_north", "east", "north", "heading_deg", "look_azimuth_deg",
        "incidence_from_vertical_deg", "grazing_angle_deg",
    )}

    unique_legs = np.unique(leg_number[in_analysis_leg & (leg_number >= 0)])
    for leg in unique_legs:
        leg_rows = np.where(in_analysis_leg & (leg_number == leg))[0]
        for row in leg_rows[::config.row_stride]:
            for side in ("port", "starboard"):
                if side == "port":
                    h, score, high, axis, gmax, n_samples = h_port[row], score_port[row], bool(high_port[row]), axis_port, gmax_port[row], len(axis_port)
                else:
                    h, score, high, axis, gmax, n_samples = h_stbd[row], score_stbd[row], bool(high_stbd[row]), axis_stbd, gmax_stbd[row], len(axis_stbd)

                if not np.isfinite(h) or not np.isfinite(gmax) or gmax <= config.min_ground_range_m + config.far_edge_margin_m:
                    continue

                ground_targets = np.arange(config.min_ground_range_m, gmax - config.far_edge_margin_m + 1e-9, config.ground_range_step_m, dtype=np.float64)
                if len(ground_targets) == 0:
                    continue

                sample_indices = nearest_axis_indices(axis, np.sqrt(ground_targets ** 2 + h ** 2))
                slant_ranges = axis[sample_indices]
                valid = slant_ranges >= h
                if not np.any(valid):
                    continue

                sample_indices, slant_ranges = sample_indices[valid], slant_ranges[valid]
                ground_ranges = np.sqrt(np.maximum(slant_ranges ** 2 - h ** 2, 0.0))
                anchor_east, anchor_north = geocode_ground_range(east[row], north[row], heading[row], ground_ranges, side)
                look_azimuth = look_azimuth_deg(heading[row], side)
                incidence = np.degrees(np.arctan2(ground_ranges, h))
                grazing = 90.0 - incidence
                raw_sample_indices = n_samples - 1 - sample_indices if side == "port" else sample_indices
                n = len(sample_indices)

                values = {
                    "survey": [survey_name] * n, "row_index": [int(row_index[row])] * n, "ping_index": [int(ping_index[row])] * n,
                    "leg_number": [int(leg)] * n, "side": [side] * n, "canonical_sample_index": sample_indices.astype(int).tolist(),
                    "raw_sample_index": raw_sample_indices.astype(int).tolist(), "slant_range_m": slant_ranges.tolist(),
                    "ground_range_m": ground_ranges.tolist(), "first_bottom_m": [float(h)] * n, "first_bottom_score": [float(score)] * n,
                    "high_confidence": [high] * n, "sensor_east": [float(east[row])] * n, "sensor_north": [float(north[row])] * n,
                    "east": anchor_east.tolist(), "north": anchor_north.tolist(), "heading_deg": [float(heading[row])] * n,
                    "look_azimuth_deg": [float(look_azimuth)] * n, "incidence_from_vertical_deg": incidence.tolist(),
                    "grazing_angle_deg": grazing.tolist(),
                }
                for key, value in values.items():
                    columns[key].extend(value)

    anchors = pd.DataFrame(columns)
    if anchors.empty:
        raise RuntimeError(f"No anchor generated for {survey_name}.")
    anchors.insert(0, "anchor_id", np.arange(len(anchors), dtype=np.int64))
    return anchors, utm_epsg


def anchor_arrays(anchors, utm_epsg, config=AnchorConfig()):
    """Serialize selected anchor fields with CRS and sampling metadata."""
    return {
        "anchor_id": anchors["anchor_id"].to_numpy(np.int64),
        "row_index": anchors["row_index"].to_numpy(np.int32),
        "ping_index": anchors["ping_index"].to_numpy(np.int64),
        "leg_number": anchors["leg_number"].to_numpy(np.int32),
        "side": anchors["side"].to_numpy(dtype="U9"),
        "canonical_sample_index": anchors["canonical_sample_index"].to_numpy(np.int32),
        "raw_sample_index": anchors["raw_sample_index"].to_numpy(np.int32),
        "slant_range_m": anchors["slant_range_m"].to_numpy(np.float64),
        "ground_range_m": anchors["ground_range_m"].to_numpy(np.float64),
        "first_bottom_m": anchors["first_bottom_m"].to_numpy(np.float64),
        "first_bottom_score": anchors["first_bottom_score"].to_numpy(np.float64),
        "high_confidence": anchors["high_confidence"].to_numpy(bool),
        "sensor_east": anchors["sensor_east"].to_numpy(np.float64),
        "sensor_north": anchors["sensor_north"].to_numpy(np.float64),
        "east": anchors["east"].to_numpy(np.float64),
        "north": anchors["north"].to_numpy(np.float64),
        "heading_deg": anchors["heading_deg"].to_numpy(np.float64),
        "look_azimuth_deg": anchors["look_azimuth_deg"].to_numpy(np.float64),
        "incidence_from_vertical_deg": anchors["incidence_from_vertical_deg"].to_numpy(np.float64),
        "grazing_angle_deg": anchors["grazing_angle_deg"].to_numpy(np.float64),
        "utm_epsg": np.asarray(utm_epsg, dtype=np.int32),
        "row_stride": np.asarray(config.row_stride, dtype=np.int32),
        "ground_range_step_m": np.asarray(config.ground_range_step_m, dtype=np.float64),
        "min_ground_range_m": np.asarray(config.min_ground_range_m, dtype=np.float64),
        "far_edge_margin_m": np.asarray(config.far_edge_margin_m, dtype=np.float64),
    }


def mutual_nearest_matches(anchors_a, anchors_b):
    """Retain anchor correspondences that are nearest neighbors in both directions."""
    points_a = anchors_a[["east", "north"]].to_numpy(dtype=np.float64)
    points_b = anchors_b[["east", "north"]].to_numpy(dtype=np.float64)
    tree_a, tree_b = cKDTree(points_a), cKDTree(points_b)
    distance_a_to_b, index_b_for_a = tree_b.query(points_a, k=1, workers=-1)
    _, index_a_for_b = tree_a.query(points_b, k=1, workers=-1)
    a_indices = np.arange(len(anchors_a), dtype=np.int64)
    mutual = index_a_for_b[index_b_for_a] == a_indices
    return a_indices[mutual], index_b_for_a[mutual].astype(np.int64), distance_a_to_b[mutual]


def build_candidate_table(survey_a, survey_b, anchors_a, anchors_b, config=AnchorConfig()):
    """Enumerate cross-pass anchor candidates and their geometry discrepancies."""
    idx_a, idx_b, distances = mutual_nearest_matches(anchors_a, anchors_b)
    within = distances <= config.max_candidate_distance_m
    a = anchors_a.iloc[idx_a[within]].reset_index(drop=True)
    b = anchors_b.iloc[idx_b[within]].reset_index(drop=True)
    candidates = pd.DataFrame({
        "match_id": np.arange(len(a), dtype=np.int64), "survey_a": survey_a, "survey_b": survey_b,
        "distance_m": distances[within], "midpoint_east": (a["east"].to_numpy() + b["east"].to_numpy()) / 2.0,
        "midpoint_north": (a["north"].to_numpy() + b["north"].to_numpy()) / 2.0,
    })

    fields = (
        "anchor_id", "row_index", "ping_index", "leg_number", "side", "canonical_sample_index", "raw_sample_index",
        "slant_range_m", "ground_range_m", "first_bottom_m", "first_bottom_score", "high_confidence",
        "sensor_east", "sensor_north", "east", "north", "heading_deg", "look_azimuth_deg",
        "incidence_from_vertical_deg", "grazing_angle_deg",
    )
    for field in fields:
        candidates[f"a_{field}"] = a[field].to_numpy()
        candidates[f"b_{field}"] = b[field].to_numpy()

    candidates["heading_difference_deg"] = angle_difference_deg(candidates["a_heading_deg"], candidates["b_heading_deg"])
    candidates["look_difference_deg"] = angle_difference_deg(candidates["a_look_azimuth_deg"], candidates["b_look_azimuth_deg"])
    candidates["incidence_difference_deg"] = np.abs(candidates["a_incidence_from_vertical_deg"] - candidates["b_incidence_from_vertical_deg"])
    candidates["both_high_confidence"] = candidates["a_high_confidence"].astype(bool) & candidates["b_high_confidence"].astype(bool)
    candidates["side_pair"] = candidates["a_side"].astype(str) + "-" + candidates["b_side"].astype(str)
    candidates["leg_pair"] = "L" + candidates["a_leg_number"].astype(str) + "-L" + candidates["b_leg_number"].astype(str)
    candidates = candidates.sort_values(["distance_m", "match_id"], kind="stable").reset_index(drop=True)
    candidates["match_id"] = np.arange(len(candidates), dtype=np.int64)
    return candidates, distances


def summarize_positive_pair(survey_a, survey_b, anchors_a, anchors_b, candidates, mutual_distances, config=AnchorConfig()):
    """Summarize matching candidates for two comparable survey acquisitions."""
    row = {
        "pair": f"{survey_a}-{survey_b}", "pair_type": "positive", "anchors_a": len(anchors_a), "anchors_b": len(anchors_b),
        "mutual_nearest_pairs": len(mutual_distances), "candidate_distance_limit_m": config.max_candidate_distance_m,
        "candidates": len(candidates), "high_confidence_candidates": int(candidates["both_high_confidence"].sum()) if not candidates.empty else 0,
        "min_mutual_distance_m": float(np.min(mutual_distances)) if len(mutual_distances) else np.nan,
        "median_mutual_distance_m": float(np.median(mutual_distances)) if len(mutual_distances) else np.nan,
    }
    for threshold in config.distance_thresholds_m:
        row[f"mutual_within_{threshold:g}m"] = int(np.count_nonzero(mutual_distances <= threshold))

    if candidates.empty:
        row.update(median_candidate_distance_m=np.nan, median_heading_difference_deg=np.nan, median_look_difference_deg=np.nan)
    else:
        high = candidates[candidates["both_high_confidence"]]
        row["median_candidate_distance_m"] = float(candidates["distance_m"].median())
        row["median_heading_difference_deg"] = float(high["heading_difference_deg"].median()) if not high.empty else np.nan
        row["median_look_difference_deg"] = float(high["look_difference_deg"].median()) if not high.empty else np.nan
    return row


def summarize_control_pair(survey_a, survey_b, anchors_a, anchors_b, config=AnchorConfig()):
    """Summarize geometric separation between an isolated control and a pass."""
    distances, _ = cKDTree(anchors_b[["east", "north"]].to_numpy(dtype=np.float64)).query(
        anchors_a[["east", "north"]].to_numpy(dtype=np.float64), k=1, workers=-1
    )
    return {
        "pair": f"{survey_a}-{survey_b}", "pair_type": "control", "anchors_a": len(anchors_a), "anchors_b": len(anchors_b),
        "mutual_nearest_pairs": np.nan, "candidate_distance_limit_m": config.max_candidate_distance_m,
        "candidates": int(np.count_nonzero(distances <= config.max_candidate_distance_m)), "high_confidence_candidates": np.nan,
        "min_mutual_distance_m": float(np.min(distances)), "median_mutual_distance_m": float(np.median(distances)),
        "median_candidate_distance_m": np.nan, "median_heading_difference_deg": np.nan, "median_look_difference_deg": np.nan,
        **{f"mutual_within_{threshold:g}m": int(np.count_nonzero(distances <= threshold)) for threshold in config.distance_thresholds_m},
    }


@dataclass(frozen=True)
class PositivePatchConfig:
    """Raster dimensions, minimum coverage and sampling for positive mosaics."""
    patch_size_m: float = 50.0
    grid_resolution_m: float = 0.25
    max_candidate_distance_m: float = 2.0
    min_ground_range_m: float = 1.0
    center_spacing_m: float = 25.0
    density_radius_m: float = 30.0
    min_coverage_each: float = 0.60
    min_joint_coverage: float = 0.50
    source_row_stride: int = 2
    source_sample_stride: int = 2
    sensor_search_extra_m: float = 30.0

    @property
    def grid_size(self):
        """Return the number of pixels on each side of a square patch."""
        return int(round(self.patch_size_m / self.grid_resolution_m))


def bool_mask(series):
    """Convert a possibly nullable table column into a boolean selection mask."""
    if series.dtype == bool:
        return series.to_numpy()
    return series.astype(str).str.lower().isin(("true", "1", "yes")).to_numpy()


def view_class(angle):
    """Classify relative viewing geometry from a look-direction angle."""
    if angle <= 20.0:
        return "similar-view"
    if angle >= 160.0:
        return "opposite-view"
    return "oblique-view"


def canonical_image(survey, side):
    """Return the selected side-scan channel in canonical sample orientation."""
    if side == "port":
        return np.flip(survey.port, axis=1)
    if side == "starboard":
        return survey.starboard
    raise ValueError(f"Invalid side: {side}")


def preselect_centres(df, config=PositivePatchConfig()):
    """Choose spatially separated matching candidate centres for paired patches."""
    pool = df.loc[
        (df["distance_m"] <= config.max_candidate_distance_m)
        & (df["a_ground_range_m"] >= config.min_ground_range_m)
        & (df["b_ground_range_m"] >= config.min_ground_range_m)
        & bool_mask(df["both_high_confidence"])
    ].copy()
    if pool.empty:
        return pool

    pool["view_class"] = pool["look_difference_deg"].map(view_class)
    xy = pool[["midpoint_east", "midpoint_north"]].to_numpy(float)
    tree = cKDTree(xy)
    pool["local_density"] = np.fromiter(
        (len(ids) for ids in tree.query_ball_point(xy, r=config.density_radius_m)),
        dtype=int,
        count=len(pool),
    )
    pool = pool.sort_values(["local_density", "distance_m", "match_id"], ascending=[False, True, True])

    selected = []
    for row in pool.itertuples(index=False):
        if selected and min(np.hypot(row.midpoint_east - s.midpoint_east, row.midpoint_north - s.midpoint_north) for s in selected) < config.center_spacing_m:
            continue
        selected.append(row)

    return pd.DataFrame([r._asdict() for r in selected])


def build_mosaic(survey, geom, center_e, center_n, config=PositivePatchConfig()):
    """Accumulate geocoded side-scan samples into one mask-aware raster patch."""
    half = config.patch_size_m / 2.0
    e_min, n_min = center_e - half, center_n - half
    grid_size = config.grid_size
    sum_grid = np.zeros((grid_size, grid_size), dtype=np.float64)
    count_grid = np.zeros((grid_size, grid_size), dtype=np.uint32)

    search_radius = np.sqrt(2.0) * half + config.sensor_search_extra_m
    sensor_dist = np.hypot(geom["east"] - center_e, geom["north"] - center_n)
    base_rows = geom["in_analysis_leg"].astype(bool) & (sensor_dist <= search_radius)

    for side, suffix in (("port", "port"), ("starboard", "stbd")):
        rows = np.flatnonzero(base_rows & geom[f"high_confidence_{suffix}"].astype(bool))[::config.source_row_stride]
        if rows.size == 0:
            continue

        image = canonical_image(survey, side)
        slant_axis = geom[f"canonical_slant_axis_{suffix}"].astype(np.float64, copy=False)
        sample_idx = np.arange(0, slant_axis.size, config.source_sample_stride, dtype=np.int32)
        slant = slant_axis[sample_idx][None, :]

        h = geom[f"first_bottom_{suffix}"][rows].astype(np.float64, copy=False)[:, None]
        valid = np.isfinite(h) & (slant >= h)
        ground = np.sqrt(np.maximum(slant * slant - h * h, 0.0))
        valid &= ground >= config.min_ground_range_m

        theta = np.deg2rad(geom["heading_deg"][rows].astype(np.float64, copy=False))[:, None]
        if side == "starboard":
            de, dn = np.cos(theta), -np.sin(theta)
        else:
            de, dn = -np.cos(theta), np.sin(theta)

        east = geom["east"][rows, None] + ground * de
        north = geom["north"][rows, None] + ground * dn
        inside = valid & (east >= e_min) & (east < e_min + config.patch_size_m) & (north >= n_min) & (north < n_min + config.patch_size_m)
        if not np.any(inside):
            continue

        values = np.log1p(image[np.ix_(rows, sample_idx)].astype(np.float64))
        ix = np.floor((east - e_min) / config.grid_resolution_m).astype(np.int32)
        iy = np.floor((north - n_min) / config.grid_resolution_m).astype(np.int32)
        linear = (iy[inside] * grid_size + ix[inside]).astype(np.int64)
        vals = values[inside]
        n_cells = grid_size * grid_size
        # Flattened cell IDs let bincount accumulate all samples without a pixel loop.
        sum_grid += np.bincount(linear, weights=vals, minlength=n_cells).reshape(grid_size, grid_size)
        count_grid += np.bincount(linear, minlength=n_cells).reshape(grid_size, grid_size).astype(np.uint32)

    mask = count_grid > 0
    mosaic = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
    mosaic[mask] = (sum_grid[mask] / count_grid[mask]).astype(np.float32)
    return mosaic, count_grid, mask


@dataclass(frozen=True)
class NullPatchConfig:
    """Parameters for bank mosaics and geometry-matched negative sampling."""
    patch_size_m: float = 50.0
    grid_resolution_m: float = 0.25
    min_ground_range_m: float = 1.0
    min_coverage: float = 0.60
    bank_center_spacing_m: float = 25.0
    max_bank_patches_per_survey: int = 24
    max_bank_candidates: int = 96
    source_row_stride: int = 2
    source_sample_stride: int = 2
    raster_row_chunk: int = 256
    global_margin_m: float = 60.0
    noncolocated_min_center_distance_m: float = 75.0
    negatives_per_positive: int = 4
    control_pairs_per_survey: int = 24

    @property
    def grid_size(self):
        """Return the number of pixels on each side of a square patch."""
        return int(round(self.patch_size_m / self.grid_resolution_m))


def select_anchor_centres(anchors, config=NullPatchConfig()):
    """Downselect spatially dispersed high-confidence anchor centres."""
    pool = anchors.loc[anchors["high_confidence"].astype(bool)].copy()
    if pool.empty:
        return pool

    cell = config.bank_center_spacing_m / 2.0
    min_e, min_n = float(pool["east"].min()), float(pool["north"].min())
    gx = np.floor((pool["east"].to_numpy(float) - min_e) / cell).astype(np.int32)
    gy = np.floor((pool["north"].to_numpy(float) - min_n) / cell).astype(np.int32)
    cx, cy = min_e + (gx + 0.5) * cell, min_n + (gy + 0.5) * cell
    pool["grid_x"], pool["grid_y"] = gx, gy
    pool["cell_center_distance"] = np.hypot(pool["east"].to_numpy(float) - cx, pool["north"].to_numpy(float) - cy)
    pool = pool.sort_values(["cell_center_distance", "anchor_id"]).drop_duplicates(["grid_x", "grid_y"])

    selected = []
    for row in pool.sort_values(["grid_y", "grid_x", "anchor_id"]).itertuples(index=False):
        if selected:
            dist = np.hypot(
                row.east - np.fromiter((s.east for s in selected), dtype=float),
                row.north - np.fromiter((s.north for s in selected), dtype=float),
            )
            if np.any(dist < config.bank_center_spacing_m):
                continue
        selected.append(row)
        if len(selected) >= config.max_bank_candidates:
            break

    return pd.DataFrame([row._asdict() for row in selected])


def _accumulate_sparse(sum_flat, count_flat, linear, values):
    """Aggregate values/counts into occupied raster cells without dense scatter loops."""
    if linear.size == 0:
        return
    unique, inverse = np.unique(linear, return_inverse=True)
    sum_flat[unique] += np.bincount(inverse, weights=values)
    count_flat[unique] += np.bincount(inverse).astype(np.uint32)


def build_global_mosaic(survey, geom, survey_name, config=NullPatchConfig()):
    """Geocode entire approved survey coverage into a sparse global mosaic."""
    analysis = geom["in_analysis_leg"].astype(bool)
    if not np.any(analysis):
        raise RuntimeError(f"No analysis rows for {survey_name}.")

    e = geom["east"][analysis].astype(np.float64, copy=False)
    n = geom["north"][analysis].astype(np.float64, copy=False)
    origin_e = np.floor((float(e.min()) - config.global_margin_m) / config.grid_resolution_m) * config.grid_resolution_m
    origin_n = np.floor((float(n.min()) - config.global_margin_m) / config.grid_resolution_m) * config.grid_resolution_m
    max_e = np.ceil((float(e.max()) + config.global_margin_m) / config.grid_resolution_m) * config.grid_resolution_m
    max_n = np.ceil((float(n.max()) + config.global_margin_m) / config.grid_resolution_m) * config.grid_resolution_m
    width = int(np.ceil((max_e - origin_e) / config.grid_resolution_m))
    height = int(np.ceil((max_n - origin_n) / config.grid_resolution_m))

    sum_grid = np.zeros((height, width), dtype=np.float64)
    count_grid = np.zeros((height, width), dtype=np.uint32)
    sum_flat, count_flat = sum_grid.ravel(), count_grid.ravel()

    for side, suffix in (("port", "port"), ("starboard", "stbd")):
        rows = np.flatnonzero(analysis & geom[f"high_confidence_{suffix}"].astype(bool))[::config.source_row_stride]
        if rows.size == 0:
            continue

        image = canonical_image(survey, side)
        slant_axis = geom[f"canonical_slant_axis_{suffix}"].astype(np.float64, copy=False)
        sample_idx = np.arange(0, slant_axis.size, config.source_sample_stride, dtype=np.int32)
        slant = slant_axis[sample_idx][None, :]

        for chunk_start in range(0, len(rows), config.raster_row_chunk):
            chunk_rows = rows[chunk_start:chunk_start + config.raster_row_chunk]
            h = geom[f"first_bottom_{suffix}"][chunk_rows].astype(np.float64, copy=False)[:, None]
            valid = np.isfinite(h) & (slant >= h)
            ground = np.sqrt(np.maximum(slant * slant - h * h, 0.0))
            valid &= ground >= config.min_ground_range_m

            theta = np.deg2rad(geom["heading_deg"][chunk_rows].astype(np.float64, copy=False))[:, None]
            if side == "starboard":
                de, dn = np.cos(theta), -np.sin(theta)
            else:
                de, dn = -np.cos(theta), np.sin(theta)

            east = geom["east"][chunk_rows, None] + ground * de
            north = geom["north"][chunk_rows, None] + ground * dn
            ix = np.floor((east - origin_e) / config.grid_resolution_m).astype(np.int32)
            iy = np.floor((north - origin_n) / config.grid_resolution_m).astype(np.int32)
            inside = valid & (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
            if np.any(inside):
                values = np.log1p(image[np.ix_(chunk_rows, sample_idx)].astype(np.float32))
                linear = iy[inside].astype(np.int64) * width + ix[inside].astype(np.int64)
                _accumulate_sparse(sum_flat, count_flat, linear, values[inside].astype(np.float64, copy=False))

    mask = count_grid > 0
    mosaic = np.full((height, width), np.nan, dtype=np.float32)
    mosaic[mask] = (sum_grid[mask] / count_grid[mask]).astype(np.float32)
    return mosaic, count_grid, origin_e, origin_n


def crop_global_patch(mosaic, count, origin_e, origin_n, center_e, center_n, config=NullPatchConfig()):
    """Crop a fixed metric footprint, retaining missing-data cells and counts."""
    half, grid_size = config.patch_size_m / 2.0, config.grid_size
    x0 = int(np.floor((center_e - half - origin_e) / config.grid_resolution_m))
    y0 = int(np.floor((center_n - half - origin_n) / config.grid_resolution_m))
    x1, y1 = x0 + grid_size, y0 + grid_size
    patch = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
    patch_count = np.zeros((grid_size, grid_size), dtype=np.uint32)

    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(mosaic.shape[1], x1), min(mosaic.shape[0], y1)
    if sx0 < sx1 and sy0 < sy1:
        dx0, dy0 = sx0 - x0, sy0 - y0
        dx1, dy1 = dx0 + (sx1 - sx0), dy0 + (sy1 - sy0)
        patch[dy0:dy1, dx0:dx1] = mosaic[sy0:sy1, sx0:sx1]
        patch_count[dy0:dy1, dx0:dx1] = count[sy0:sy1, sx0:sx1]

    return patch, patch_count, patch_count > 0


def build_patch_bank(survey, geom, anchors, survey_name, config=NullPatchConfig()):
    """Create non-colocated candidate patches and per-observation metadata."""
    centres = select_anchor_centres(anchors, config)
    mosaic, mosaic_count, origin_e, origin_n = build_global_mosaic(survey, geom, survey_name, config)
    records, patches, masks, counts = [], [], [], []

    for row in centres.itertuples(index=False):
        patch, count, mask = crop_global_patch(mosaic, mosaic_count, origin_e, origin_n, float(row.east), float(row.north), config)
        coverage = float(mask.mean())
        if coverage < config.min_coverage:
            continue

        record = row._asdict()
        record.update(survey=survey_name, bank_index=len(records), coverage=coverage)
        records.append(record)
        patches.append(patch)
        masks.append(mask)
        counts.append(count)
        if len(records) >= config.max_bank_patches_per_survey:
            break

    if not records:
        raise RuntimeError(f"No valid standalone patches for {survey_name}.")

    metadata = pd.DataFrame(records)
    arrays = {
        "patch": np.stack(patches).astype(np.float32),
        "mask": np.stack(masks).astype(bool),
        "count": np.stack(counts).astype(np.uint32),
        "bank_index": metadata["bank_index"].to_numpy(np.int32),
        "anchor_id": metadata["anchor_id"].to_numpy(np.int64),
        "center_east": metadata["east"].to_numpy(float),
        "center_north": metadata["north"].to_numpy(float),
        "coverage": metadata["coverage"].to_numpy(float),
        "side": metadata["side"].astype(str).to_numpy(),
        "heading_deg": metadata["heading_deg"].to_numpy(float),
        "look_azimuth_deg": metadata["look_azimuth_deg"].to_numpy(float),
        "incidence_from_vertical_deg": metadata["incidence_from_vertical_deg"].to_numpy(float),
        "patch_size_m": np.float64(config.patch_size_m),
        "grid_resolution_m": np.float64(config.grid_resolution_m),
        "representation": np.asarray("geocoded_log1p_mean"),
    }
    return metadata, arrays


def pair_geometry(bank_a, bank_b):
    """Measure distances and relative heading/look/incidence between bank patches."""
    rows = []
    for a in bank_a.itertuples(index=False):
        for b in bank_b.itertuples(index=False):
            rows.append({
                "bank_index_a": int(a.bank_index), "bank_index_b": int(b.bank_index),
                "center_east_a": float(a.east), "center_north_a": float(a.north),
                "center_east_b": float(b.east), "center_north_b": float(b.north),
                "coverage_a": float(a.coverage), "coverage_b": float(b.coverage),
                "center_separation_m": float(np.hypot(a.east - b.east, a.north - b.north)),
                "heading_difference_deg": float(angle_difference_deg(a.heading_deg, b.heading_deg)),
                "look_difference_deg": float(angle_difference_deg(a.look_azimuth_deg, b.look_azimuth_deg)),
                "incidence_difference_deg": float(abs(a.incidence_from_vertical_deg - b.incidence_from_vertical_deg)),
            })
    return pd.DataFrame(rows)


def select_within_site_negatives(name_a, name_b, bank_a, bank_b, positives, config=NullPatchConfig()):
    """Assign distant bank pairs to positives by similarity of acquisition geometry."""
    combos = pair_geometry(bank_a, bank_b)
    combos = combos.loc[combos["center_separation_m"] >= config.noncolocated_min_center_distance_m].copy()
    if combos.empty:
        raise RuntimeError(f"No disjoint negative combinations for {name_a}-{name_b}.")

    used, selected = set(), []
    for pos in positives.itertuples(index=False):
        score = (
            angle_difference_deg(combos["heading_difference_deg"], pos.heading_difference_deg) / 20.0
            + angle_difference_deg(combos["look_difference_deg"], pos.look_difference_deg) / 20.0
            + np.abs(combos["incidence_difference_deg"] - pos.incidence_difference_deg) / 20.0
        )
        for idx in np.argsort(score.to_numpy()):
            row = combos.iloc[int(idx)]
            key = (int(row.bank_index_a), int(row.bank_index_b))
            if key in used:
                continue
            used.add(key)
            record = row.to_dict()
            record.update(
                null_type="within-site-noncolocated", survey_a=name_a, survey_b=name_b,
                matched_positive_patch_id=int(pos.patch_id), matched_positive_view_class=str(pos.view_class),
                geometry_match_score=float(score.iloc[int(idx)]),
            )
            selected.append(record)
            if sum(r["matched_positive_patch_id"] == int(pos.patch_id) for r in selected) >= config.negatives_per_positive:
                break

    return pd.DataFrame(selected)


def select_control_negatives(target_name, control_bank, target_bank, n_pairs):
    """Sample a balanced isolated-control set from two patch banks."""
    combos = pair_geometry(control_bank, target_bank)
    if combos.empty:
        raise RuntimeError(f"No CONTROL combinations available for {target_name}.")

    control_usage = {int(i): 0 for i in control_bank["bank_index"]}
    target_usage = {int(i): 0 for i in target_bank["bank_index"]}
    available, selected = combos.copy(), []

    for _ in range(min(n_pairs, len(available))):
        c_use = available["bank_index_a"].map(control_usage).to_numpy(np.int32)
        t_use = available["bank_index_b"].map(target_usage).to_numpy(np.int32)
        coverage_score = (available["coverage_a"] + available["coverage_b"]).to_numpy(float)
        # lexsort uses the last key first: balance control usage, then target usage,
        # and prefer higher joint coverage when usage counts tie.
        idx = int(np.lexsort((-coverage_score, t_use, c_use))[0])
        row = available.iloc[idx]

        c_idx, t_idx = int(row.bank_index_a), int(row.bank_index_b)
        control_usage[c_idx] += 1
        target_usage[t_idx] += 1

        record = row.to_dict()
        record.update(
            null_type="isolated-control", survey_a="control", survey_b=target_name,
            matched_positive_patch_id=-1, matched_positive_view_class="control", geometry_match_score=np.nan,
        )
        selected.append(record)
        available = available.drop(available.index[idx]).reset_index(drop=True)

    return pd.DataFrame(selected)
