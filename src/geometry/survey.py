"""Recover usable, approximately straight survey legs from ping navigation.

Input: ping coordinates, timestamps and headings for each survey.
Output: projected UTM tracks and candidate straight-leg metadata.
Usage: imported by DatasetBuilder; no standalone CLI."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pyproj import Transformer


@dataclass(frozen=True)
class LegDetectionConfig:
    """Thresholds used to identify continuous, sufficiently straight survey legs."""
    stride: int = 20
    course_half_window: int = 5
    max_local_turn_deg: float = 8.0
    min_speed_m_s: float = 0.25
    max_gap_seconds: float = 5.0
    min_duration_seconds: float = 20.0
    min_displacement_m: float = 25.0
    min_straightness: float = 0.90


def angle_difference_deg(angle_a, angle_b):
    """Return the minimal circular difference between headings in degrees."""
    difference = abs(angle_a - angle_b) % 360.0
    return min(difference, 360.0 - difference)


def vector_to_course_deg(delta_east, delta_north):
    """Convert east/north displacement vectors to navigation course angles."""
    return float(np.degrees(np.arctan2(delta_east, delta_north)) % 360.0)


def circular_mean_deg(angles_deg):
    """Average angles without incorrectly wrapping across north (0/360 degrees)."""
    angles_rad = np.deg2rad(angles_deg)
    return float(np.degrees(np.arctan2(np.mean(np.sin(angles_rad)), np.mean(np.cos(angles_rad)))) % 360.0)


def infer_utm_epsg(tracks):
    """Choose a UTM CRS appropriate to the recorded survey coordinates."""
    all_lon = np.concatenate([track["x"].to_numpy() for track in tracks.values()])
    all_lat = np.concatenate([track["y"].to_numpy() for track in tracks.values()])
    reference_lon, reference_lat = float(np.mean(all_lon)), float(np.mean(all_lat))
    utm_zone = int((reference_lon + 180) // 6) + 1
    return (32600 if reference_lat >= 0 else 32700) + utm_zone


def project_tracks_to_utm(tracks):
    """Transform lon/lat tracks to metric UTM east/north coordinates."""
    epsg = infer_utm_epsg(tracks)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    projected = {}

    for name, track in tracks.items():
        track = track.copy()
        track["east"], track["north"] = transformer.transform(track["x"].to_numpy(), track["y"].to_numpy())
        projected[name] = track

    return projected, epsg


def detect_survey_legs(track, config=LegDetectionConfig()):
    """Segment tracks into straight, long-enough, temporally continuous legs."""
    dt = track["timestamp"].diff().dt.total_seconds().dropna()
    median_ping_dt = float(dt.median())
    sampled = track.iloc[::config.stride].copy().reset_index(drop=True)
    sampled["original_row"] = np.arange(0, len(track), config.stride)[:len(sampled)]

    if len(sampled) < 2 * config.course_half_window + 1:
        return [], sampled

    sampled_dt = median_ping_dt * config.stride
    local_turn = np.full(len(sampled), np.nan)
    local_speed = np.full(len(sampled), np.nan)

    for i in range(config.course_half_window, len(sampled) - config.course_half_window):
        previous, current, following = sampled.iloc[i - config.course_half_window], sampled.iloc[i], sampled.iloc[i + config.course_half_window]
        back_east, back_north = current["east"] - previous["east"], current["north"] - previous["north"]
        forward_east, forward_north = following["east"] - current["east"], following["north"] - current["north"]
        back_distance, forward_distance = np.hypot(back_east, back_north), np.hypot(forward_east, forward_north)
        local_turn[i] = angle_difference_deg(vector_to_course_deg(back_east, back_north), vector_to_course_deg(forward_east, forward_north))
        local_speed[i] = (back_distance + forward_distance) / (2.0 * config.course_half_window * sampled_dt)

    sampled["local_turn_deg"] = local_turn
    sampled["local_speed_m_s"] = local_speed
    candidate = ((sampled["local_turn_deg"] <= config.max_local_turn_deg) & (sampled["local_speed_m_s"] >= config.min_speed_m_s)).to_numpy(dtype=bool, copy=True)

    max_gap_points = max(1, int(round(config.max_gap_seconds / sampled_dt)))
    i = 0
    while i < len(candidate):
        if candidate[i]:
            i += 1
            continue
        gap_start = i
        while i < len(candidate) and not candidate[i]:
            i += 1
        if gap_start > 0 and i < len(candidate) and candidate[gap_start - 1] and candidate[i] and i - gap_start <= max_gap_points:
            candidate[gap_start:i] = True

    sampled["straight_candidate"] = candidate
    raw_segments = []
    i = 0
    while i < len(sampled):
        if not candidate[i]:
            i += 1
            continue
        start = i
        while i + 1 < len(sampled) and candidate[i + 1]:
            i += 1
        raw_segments.append((start, i))
        i += 1

    legs = []
    for start, end in raw_segments:
        segment = sampled.iloc[start:end + 1]
        duration = (segment["timestamp"].iloc[-1] - segment["timestamp"].iloc[0]).total_seconds()
        delta_east = float(segment["east"].iloc[-1] - segment["east"].iloc[0])
        delta_north = float(segment["north"].iloc[-1] - segment["north"].iloc[0])
        displacement = float(np.hypot(delta_east, delta_north))
        segment_east, segment_north = segment["east"].to_numpy(), segment["north"].to_numpy()
        path_length = float(np.sum(np.hypot(np.diff(segment_east), np.diff(segment_north))))
        straightness = displacement / path_length if path_length > 0 else 0.0

        if duration < config.min_duration_seconds or displacement < config.min_displacement_m or straightness < config.min_straightness:
            continue

        course = vector_to_course_deg(delta_east, delta_north)
        start_row, end_row = int(segment["original_row"].iloc[0]), int(segment["original_row"].iloc[-1])
        legs.append({
            "start_sampled": start, "end_sampled": end, "start_row": start_row, "end_row": end_row,
            "start_ping": int(track["ping_index"].iloc[start_row]), "end_ping": int(track["ping_index"].iloc[end_row]),
            "start_timestamp": track["timestamp"].iloc[start_row], "end_timestamp": track["timestamp"].iloc[end_row],
            "duration_s": duration, "displacement_m": displacement, "path_length_m": path_length,
            "straightness": straightness, "course_deg": course, "line_orientation_deg": course % 180.0,
            "raw_heading_mean_deg": circular_mean_deg(segment["heading_raw"].to_numpy()),
        })

    return legs, sampled
