"""Estimate first-bottom arrival and geocode side-scan ranges to seabed coordinates.

Input: XTFSurvey, projected navigation, survey-leg metadata and tracking arrays.
Output: per-side bottom estimates and geometric arrays consumed by patch extraction.
Usage: imported by DatasetBuilder; no standalone CLI."""

from dataclasses import dataclass, field

import numpy as np
from pyproj import Transformer
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True)
class FirstBottomConfig:
    """Parameterize first-bottom search and temporal path smoothing."""
    block_size: int = 200
    block_step: int = 100
    search_min_range_m: float = 0.4
    search_max_range_m: float = 10.0
    min_segment_width_m: float = 0.4
    smooth_sigma: float = 3.0
    candidate_stride: int = 2
    transition_weight: float = 0.05
    transition_scale_m: float = 0.50


@dataclass(frozen=True)
class SeabedGeometryConfig:
    """Define confidence criteria and selected analysis-leg ranges."""
    leg_edge_margin_rows: int = 100
    low_score_threshold: float = 0.30
    benchmark_legs: dict = field(default_factory=lambda: {
        "control": list(range(2, 12)), "pass_a": list(range(3, 11)),
        "pass_b": list(range(3, 12)), "pass_c": list(range(1, 5)),
    })


def canonical_slant_axis(n_samples, slant_range):
    """Map sonar sample indices to slant-range cell centres in metres."""
    return (np.arange(n_samples, dtype=np.float64) + 0.5) * float(slant_range) / n_samples


def compute_split_scores(profile, range_axis, config=FirstBottomConfig()):
    """Score likely water-column/seabed transitions along one sonar profile."""
    profile = gaussian_filter1d(np.log1p(profile.astype(np.float64)), sigma=config.smooth_sigma)
    valid_indices = np.where((range_axis >= config.search_min_range_m) & (range_axis <= config.search_max_range_m))[0]
    if len(valid_indices) == 0:
        raise RuntimeError("No samples within the search range.")

    first, last = int(valid_indices[0]), int(valid_indices[-1])
    search_profile, search_ranges = profile[first:last + 1], range_axis[first:last + 1]
    spacing = float(np.median(np.diff(search_ranges)))
    min_segment_samples = max(2, int(round(config.min_segment_width_m / spacing)))
    n = len(search_profile)
    cumulative, cumulative_sq = np.cumsum(search_profile), np.cumsum(search_profile ** 2)
    global_mean = float(np.mean(search_profile))
    sse_single = float(np.sum((search_profile - global_mean) ** 2))
    split_positions = np.arange(min_segment_samples, n - min_segment_samples, config.candidate_stride, dtype=int)
    if len(split_positions) == 0:
        raise RuntimeError("Profile is too short to build candidates.")

    candidate_ranges = search_ranges[split_positions]
    scores = np.zeros(len(split_positions), dtype=np.float64)
    for candidate_index, split in enumerate(split_positions):
        n_left, n_right = split, n - split
        sum_left, sum_sq_left = cumulative[split - 1], cumulative_sq[split - 1]
        sum_right = cumulative[-1] - cumulative[split - 1]
        sum_sq_right = cumulative_sq[-1] - cumulative_sq[split - 1]
        mean_left, mean_right = sum_left / n_left, sum_right / n_right
        if mean_right <= mean_left:
            continue
        sse_left = sum_sq_left - sum_left ** 2 / n_left
        sse_right = sum_sq_right - sum_right ** 2 / n_right
        score = (sse_single - (sse_left + sse_right)) / sse_single if sse_single > 0 else 0.0
        scores[candidate_index] = np.clip(score, 0.0, 1.0)
    return candidate_ranges, scores


def build_score_matrix(sonar, range_axis, config=FirstBottomConfig()):
    """Compute first-bottom candidate costs along successive sonar pings."""
    block_centres, score_rows, candidate_ranges = [], [], None
    for start in range(0, sonar.shape[0] - config.block_size + 1, config.block_step):
        centre = start + config.block_size // 2
        ranges, scores = compute_split_scores(np.median(sonar[start:start + config.block_size], axis=0), range_axis, config)
        if candidate_ranges is None:
            candidate_ranges = ranges
        elif not np.allclose(candidate_ranges, ranges):
            raise RuntimeError("Inconsistent candidate range grid.")
        block_centres.append(centre)
        score_rows.append(scores)
    if not score_rows:
        raise RuntimeError("No blocks available for tracking.")
    return np.asarray(block_centres, dtype=int), candidate_ranges, np.stack(score_rows, axis=0)


def track_best_path(candidate_ranges, score_matrix, config=FirstBottomConfig()):
    """Track a temporally coherent bottom interface across candidate ranges."""
    n_blocks, n_candidates = score_matrix.shape
    # Dynamic programming balances strong bottom evidence against implausible
    # jumps between successive ping blocks (a smoother tracked bottom path).
    observation_cost = 1.0 - score_matrix
    range_difference = candidate_ranges[:, None] - candidate_ranges[None, :]
    transition_cost = config.transition_weight * (range_difference / config.transition_scale_m) ** 2
    dp_previous = observation_cost[0].copy()
    backpointer = np.zeros((n_blocks, n_candidates), dtype=np.int32)

    for t in range(1, n_blocks):
        total_cost = dp_previous[:, None] + transition_cost
        best_previous = np.argmin(total_cost, axis=0)
        dp_previous = observation_cost[t] + total_cost[best_previous, np.arange(n_candidates)]
        backpointer[t] = best_previous

    best_path_indices = np.zeros(n_blocks, dtype=np.int32)
    best_path_indices[-1] = int(np.argmin(dp_previous))
    for t in range(n_blocks - 1, 0, -1):
        best_path_indices[t - 1] = backpointer[t, best_path_indices[t]]
    return candidate_ranges[best_path_indices], score_matrix[np.arange(n_blocks), best_path_indices]


def estimate_first_bottom(survey, config=FirstBottomConfig()):
    """Estimate and temporally smooth first-bottom ranges on both sonar sides."""
    port, starboard = np.flip(survey.port, axis=1), survey.starboard
    port_axis = canonical_slant_axis(port.shape[1], survey.slant_range_port)
    stbd_axis = canonical_slant_axis(starboard.shape[1], survey.slant_range_starboard)
    block_port, ranges_port, scores_port = build_score_matrix(port, port_axis, config)
    block_stbd, ranges_stbd, scores_stbd = build_score_matrix(starboard, stbd_axis, config)
    if not np.array_equal(block_port, block_stbd):
        raise RuntimeError("PORT/STBD blocks are not aligned.")
    if np.any(block_port >= len(survey.metadata)):
        raise RuntimeError("Block center is outside the metadata DataFrame.")

    raw_port_idx, raw_stbd_idx = np.argmax(scores_port, axis=1), np.argmax(scores_stbd, axis=1)
    tracked_port, tracked_port_scores = track_best_path(ranges_port, scores_port, config)
    tracked_stbd, tracked_stbd_scores = track_best_path(ranges_stbd, scores_stbd, config)
    return {
        "block_centres": block_port,
        "block_ping_indices": survey.metadata["ping_index"].iloc[block_port].to_numpy(dtype=int),
        "raw_port_ranges": ranges_port[raw_port_idx], "raw_stbd_ranges": ranges_stbd[raw_stbd_idx],
        "raw_port_scores": scores_port[np.arange(len(block_port)), raw_port_idx],
        "raw_stbd_scores": scores_stbd[np.arange(len(block_port)), raw_stbd_idx],
        "tracked_port": tracked_port, "tracked_stbd": tracked_stbd,
        "tracked_port_scores": tracked_port_scores, "tracked_stbd_scores": tracked_stbd_scores,
        "side_difference": np.abs(tracked_port - tracked_stbd),
        "block_size": np.asarray(config.block_size, dtype=np.int32), "block_step": np.asarray(config.block_step, dtype=np.int32),
    }


def interpolate_inside(x, xp, fp):
    """Interpolate values within the observed support of a track."""
    x, xp, fp = np.asarray(x, dtype=float), np.asarray(xp, dtype=float), np.asarray(fp, dtype=float)
    result = np.full(x.shape, np.nan, dtype=np.float64)
    if len(xp) == 0:
        return result
    inside = (x >= xp[0]) & (x <= xp[-1])
    result[inside] = np.interp(x[inside], xp, fp)
    return result


def slant_to_ground(slant_range, altitude):
    """Convert slant range to horizontal seabed distance using altitude."""
    r, h = np.asarray(slant_range, dtype=np.float64), np.asarray(altitude, dtype=np.float64)
    r_b, h_b = np.broadcast_arrays(r, h)
    result = np.full(r_b.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(r_b) & np.isfinite(h_b) & (r_b >= 0.0) & (h_b >= 0.0) & (r_b >= h_b)
    
    """ Pythagorean slant-to-ground conversion: r² = h² + ground²."""
    result[valid] = np.sqrt(np.maximum(r_b[valid] ** 2 - h_b[valid] ** 2, 0.0))
    return result


def geocode_ground_range(sensor_east, sensor_north, heading_deg, ground_range, side):
    """Map sensor position, heading and side-specific ground range to UTM points."""
    theta = np.deg2rad(np.asarray(heading_deg, dtype=np.float64))
    sensor_east, sensor_north = np.asarray(sensor_east, dtype=np.float64), np.asarray(sensor_north, dtype=np.float64)
    ground_range = np.asarray(ground_range, dtype=np.float64)
    if side == "starboard":
        direction_east, direction_north = np.cos(theta), -np.sin(theta)
    elif side == "port":
        direction_east, direction_north = -np.cos(theta), np.sin(theta)
    else:
        raise ValueError("side must be 'port' or 'starboard'.")
    return sensor_east + ground_range * direction_east, sensor_north + ground_range * direction_north


def select_benchmark_legs(legs, survey_name, benchmark_legs):
    """Select only approved survey legs for benchmark geometry."""
    if survey_name not in benchmark_legs:
        raise KeyError(f"No benchmark selection defined for {survey_name}.")
    selected_numbers = list(benchmark_legs[survey_name])
    available_numbers = set(legs["leg_number"].astype(int).tolist())
    missing = [number for number in selected_numbers if number not in available_numbers]
    if missing:
        raise RuntimeError(f"Required survey legs missing for {survey_name}: " + ", ".join(f"L{n}" for n in missing))
    return legs[legs["leg_number"].isin(selected_numbers)].copy().sort_values("start_row").reset_index(drop=True)


def build_leg_arrays(legs, n_rows, survey_name, config=SeabedGeometryConfig()):
    """Mark the ping rows belonging to the selected analysis legs."""
    leg_number = np.full(n_rows, -1, dtype=np.int32)
    leg_course_deg = np.full(n_rows, np.nan, dtype=np.float64)
    leg_orientation_deg = np.full(n_rows, np.nan, dtype=np.float64)
    in_analysis_leg = np.zeros(n_rows, dtype=bool)
    used_rows = []

    for _, leg in select_benchmark_legs(legs, survey_name, config.benchmark_legs).iterrows():
        original_start, original_end = int(leg["start_row"]), int(leg["end_row"])
        start, end = max(0, original_start + config.leg_edge_margin_rows), min(n_rows - 1, original_end - config.leg_edge_margin_rows)
        if end < start:
            continue
        rows = slice(start, end + 1)
        number, course, orientation = int(leg["leg_number"]), float(leg["course_deg"]), float(leg["line_orientation_deg"])
        leg_number[rows], leg_course_deg[rows], leg_orientation_deg[rows], in_analysis_leg[rows] = number, course, orientation, True
        used_rows.append({
            "survey": survey_name, "leg_number": number, "leg_id": leg["leg_id"],
            "original_start_row": original_start, "original_end_row": original_end,
            "effective_start_row": start, "effective_end_row": end,
            "edge_margin_rows": config.leg_edge_margin_rows, "course_deg": course, "line_orientation_deg": orientation,
        })
    return leg_number, leg_course_deg, leg_orientation_deg, in_analysis_leg, used_rows


def prepare_seabed_geometry(survey, survey_name, legs_table, tracking, utm_epsg, config=SeabedGeometryConfig()):
    """Combine navigation and first-bottom tracks into per-ping ground geometry."""
    metadata = survey.metadata.copy()
    n_rows, n_samples = len(metadata), survey.port.shape[1]
    if survey.port.shape[0] != n_rows or survey.starboard.shape[0] != n_rows:
        raise RuntimeError("Metadata and SONAR matrices have different row counts.")

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    east, north = transformer.transform(metadata["x"].to_numpy(), metadata["y"].to_numpy())
    east, north = np.asarray(east, dtype=np.float64), np.asarray(north, dtype=np.float64)
    heading_deg = metadata["heading_raw"].to_numpy(dtype=np.float64)
    row_index, ping_index = np.arange(n_rows, dtype=np.int32), metadata["ping_index"].to_numpy(dtype=np.int64)

    survey_legs = legs_table[legs_table["survey"] == survey_name].copy()
    if survey_legs.empty:
        raise RuntimeError(f"No survey legs found for {survey_name}.")
    leg_number, leg_course_deg, leg_orientation_deg, in_analysis_leg, used_rows = build_leg_arrays(survey_legs, n_rows, survey_name, config)

    block_centres = tracking["block_centres"].astype(int)
    tracked_port, tracked_stbd = tracking["tracked_port"].astype(float), tracking["tracked_stbd"].astype(float)
    tracked_port_scores = tracking["tracked_port_scores"].astype(float)
    tracked_stbd_scores = tracking["tracked_stbd_scores"].astype(float)
    if np.any(block_centres < 0) or np.any(block_centres >= n_rows):
        raise RuntimeError("Step-15 block_centres are incompatible with the survey rows.")
    if hasattr(tracking, "files") and "block_ping_indices" in tracking.files:
        if not np.array_equal(ping_index[block_centres], tracking["block_ping_indices"].astype(np.int64)):
            raise RuntimeError("Step 15 and the current survey are misaligned: block_ping_indices do not match.")

    h_port = interpolate_inside(row_index, block_centres, tracked_port)
    h_stbd = interpolate_inside(row_index, block_centres, tracked_stbd)
    score_port = interpolate_inside(row_index, block_centres, tracked_port_scores)
    score_stbd = interpolate_inside(row_index, block_centres, tracked_stbd_scores)
    for array in (h_port, h_stbd, score_port, score_stbd):
        array[~in_analysis_leg] = np.nan

    low_score_port = np.isfinite(score_port) & (score_port < config.low_score_threshold)
    low_score_stbd = np.isfinite(score_stbd) & (score_stbd < config.low_score_threshold)
    port_axis, stbd_axis = canonical_slant_axis(n_samples, survey.slant_range_port), canonical_slant_axis(n_samples, survey.slant_range_starboard)
    max_port_slant, max_stbd_slant = float(port_axis[-1]), float(stbd_axis[-1])
    available_port = in_analysis_leg & np.isfinite(h_port) & (h_port >= 0.0) & (h_port <= max_port_slant)
    available_stbd = in_analysis_leg & np.isfinite(h_stbd) & (h_stbd >= 0.0) & (h_stbd <= max_stbd_slant)
    high_confidence_port, high_confidence_stbd = available_port & ~low_score_port, available_stbd & ~low_score_stbd
    gmax_port, gmax_stbd = slant_to_ground(max_port_slant, h_port), slant_to_ground(max_stbd_slant, h_stbd)
    port_far_east, port_far_north = geocode_ground_range(east, north, heading_deg, gmax_port, "port")
    stbd_far_east, stbd_far_north = geocode_ground_range(east, north, heading_deg, gmax_stbd, "starboard")

    data = {
        "survey_name": np.asarray(survey_name), "utm_epsg": np.asarray(utm_epsg, dtype=np.int32),
        "row_index": row_index, "ping_index": ping_index, "east": east, "north": north, "heading_deg": heading_deg,
        "leg_number": leg_number, "leg_course_deg": leg_course_deg, "leg_orientation_deg": leg_orientation_deg,
        "in_analysis_leg": in_analysis_leg, "first_bottom_port": h_port, "first_bottom_stbd": h_stbd,
        "first_bottom_score_port": score_port, "first_bottom_score_stbd": score_stbd,
        "low_score_port": low_score_port, "low_score_stbd": low_score_stbd,
        "first_bottom_available_port": available_port, "first_bottom_available_stbd": available_stbd,
        "high_confidence_port": high_confidence_port, "high_confidence_stbd": high_confidence_stbd,
        "canonical_slant_axis_port": port_axis, "canonical_slant_axis_stbd": stbd_axis,
        "max_ground_range_port": gmax_port, "max_ground_range_stbd": gmax_stbd,
        "port_far_east": port_far_east, "port_far_north": port_far_north,
        "stbd_far_east": stbd_far_east, "stbd_far_north": stbd_far_north,
        "leg_edge_margin_rows": np.asarray(config.leg_edge_margin_rows, dtype=np.int32),
        "low_score_threshold": np.asarray(config.low_score_threshold, dtype=np.float64),
        "benchmark_leg_numbers": np.asarray(config.benchmark_legs[survey_name], dtype=np.int32),
    }
    return data, used_rows
