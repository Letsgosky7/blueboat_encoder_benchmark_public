"""Survey navigation, first-bottom tracking and geocoding functions; import-only package.

Input: none at package import time.
Output: no files; package namespace only.
Usage: import submodules from the repository root or invoke the documented scripts.
"""

from .survey import LegDetectionConfig, angle_difference_deg, circular_mean_deg, detect_survey_legs, infer_utm_epsg, project_tracks_to_utm, vector_to_course_deg