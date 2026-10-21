"""Shared fixed-protocol constants for the BlueBoat frozen encoder benchmark.

Input: established 50 m dataset; output: shared survey/condition identifiers.
Imported by pipeline, reporting and diagnostics; no CLI.
"""

POSITIVE_PAIRS = (
    ("pass_a", "pass_b"),
    ("pass_a", "pass_c"),
    ("pass_b", "pass_c"),
)
BANK_SURVEYS = ("control", "pass_a", "pass_b", "pass_c")
CONDITIONS = ("sonar", "mask-only")
GROUPS = ("positive", "within-site-noncolocated", "isolated-control")
EXPECTED_COUNTS = (25, 81, 172)  # positives, bank observations, negatives
