"""Patch colocation and geocoded mosaic utilities; import-only package.

Input: none at package import time.
Output: no files; package namespace only.
Usage: import submodules from the repository root or invoke the documented scripts.
"""

from .extraction import AnchorConfig, angle_difference_deg, anchor_arrays, build_anchors, build_candidate_table, geocode_ground_range, look_azimuth_deg, mutual_nearest_matches, nearest_axis_indices, summarize_control_pair, summarize_positive_pair
