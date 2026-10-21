"""Assemble the canonical BlueBoat repeat-pass dataset and its manifest.

Input: four XTF files below <dataset_root>/SSS-XTF (see SURVEYS).
Output: outputs/tables/*.csv or *.npz, outputs/patches/*.npz and outputs/manifest.json.
Usage: python -m scripts.build_dataset with configs/local.yaml; this is its library implementation."""

from pathlib import Path
import gc
import json

import numpy as np
import pandas as pd

from src.data.xtf_reader import read_xtf
from src.geometry.seabed import estimate_first_bottom, prepare_seabed_geometry
from src.geometry.survey import detect_survey_legs, project_tracks_to_utm
from src.patches.extraction import (
    AnchorConfig,
    NullPatchConfig,
    PositivePatchConfig,
    anchor_arrays,
    build_anchors,
    build_candidate_table,
    build_mosaic,
    build_patch_bank,
    preselect_centres,
    select_control_negatives,
    select_within_site_negatives,
    summarize_control_pair,
    summarize_positive_pair,
)


SURVEYS = {
    "control": "2026-08-04-12-59.xtf",
    "pass_a": "2026-08-05-08-00-1.xtf",
    "pass_b": "2026-08-06-07-50.xtf",
    "pass_c": "2026-08-06-09-10.xtf",
}
POSITIVE_PAIRS = (("pass_a", "pass_b"), ("pass_a", "pass_c"), ("pass_b", "pass_c"))
CONTROL_PAIRS = (("control", "pass_a"), ("control", "pass_b"), ("control", "pass_c"))

GEOMETRY_KEYS = (
    "east", "north", "heading_deg", "in_analysis_leg", "first_bottom_port", "first_bottom_stbd",
    "high_confidence_port", "high_confidence_stbd", "canonical_slant_axis_port", "canonical_slant_axis_stbd",
)
ANCHOR_KEYS = (
    "anchor_id", "side", "east", "north", "heading_deg", "look_azimuth_deg",
    "incidence_from_vertical_deg", "high_confidence",
)


class DatasetBuilder:
    """Orchestrate geometry, candidate selection, patch creation and manifest writing."""
    def __init__(self, project_root, dataset_root):
        """Resolve data directories and create the local table/patch output folders."""
        self.project_root = Path(project_root)
        self.xtf_dir = Path(dataset_root) / "SSS-XTF"
        self.table_dir = self.project_root / "outputs" / "tables"
        self.patch_dir = self.project_root / "outputs" / "patches"
        self.table_dir.mkdir(parents=True, exist_ok=True)
        self.patch_dir.mkdir(parents=True, exist_ok=True)

    def geometry_path(self, name):
        """Return the stored geometry NPZ path for a survey."""
        return self.table_dir / f"seabed_geometry_{name}.npz"

    def anchor_path(self, name):
        """Return the stored seabed-anchor NPZ path for a survey."""
        return self.table_dir / f"seabed_anchors_{name}.npz"

    def candidate_path(self, a, b):
        """Return the colocated-candidates CSV path for a survey pair."""
        return self.table_dir / f"colocation_candidates_{a}_{b}.csv"

    def load_geometry(self, name):
        """Load the required per-survey geometry arrays and validate keys."""
        path = self.geometry_path(name)
        with np.load(path) as archive:
            missing = [key for key in GEOMETRY_KEYS if key not in archive.files]
            if missing:
                raise KeyError(f"Missing geometry arrays in {path.name}: {missing}")
            return {key: np.asarray(archive[key]) for key in GEOMETRY_KEYS}

    def load_anchors(self, name):
        """Load selected seabed anchors as a typed DataFrame."""
        path = self.anchor_path(name)
        with np.load(path) as archive:
            missing = [key for key in ANCHOR_KEYS if key not in archive.files]
            if missing:
                raise KeyError(f"Missing anchor arrays in {path.name}: {missing}")
            return pd.DataFrame({key: np.asarray(archive[key]) for key in ANCHOR_KEYS})

    def build_legs(self):
        """Detect straight survey segments, project positions to UTM and save metadata."""
        tracks = {}
        for name, filename in SURVEYS.items():
            survey = read_xtf(self.xtf_dir / filename)
            tracks[name] = survey.metadata[["ping_index", "timestamp", "x", "y", "heading_raw"]].copy()
            del survey
            gc.collect()

        tracks, epsg = project_tracks_to_utm(tracks)
        rows, counts = [], {}
        for name, track in tracks.items():
            legs, _ = detect_survey_legs(track)
            counts[name] = len(legs)
            for number, leg in enumerate(legs, start=1):
                rows.append({
                    "survey": name, "filename": SURVEYS[name], "leg_number": number,
                    "leg_id": f"{name.upper()}_L{number}", "utm_epsg": epsg, **leg,
                })

        table = pd.DataFrame(rows)
        table.to_csv(self.table_dir / "survey_legs.csv", index=False)
        return table, counts

    def build_first_bottom(self):
        """Estimate the seabed boundary on each XTF recording and cache tracks."""
        stats = {}
        for name, filename in SURVEYS.items():
            survey = read_xtf(self.xtf_dir / filename)
            data = estimate_first_bottom(survey)
            np.savez_compressed(self.table_dir / f"first_bottom_track_{name}.npz", **data)
            stats[name] = {
                "blocks": len(data["block_centres"]),
                "port": float(np.median(data["tracked_port"])),
                "stbd": float(np.median(data["tracked_stbd"])),
                "side_diff": float(np.median(data["side_difference"])),
            }
            del survey, data
            gc.collect()
        return stats

    def build_geometry(self):
        """Combine navigation, straight legs and first-bottom estimates into geocoded geometry."""
        legs = pd.read_csv(self.table_dir / "survey_legs.csv")
        epsg_values = legs["utm_epsg"].dropna().astype(int).unique()
        if len(epsg_values) != 1:
            raise RuntimeError("survey_legs.csv must use a single UTM EPSG code.")
        utm_epsg = int(epsg_values[0])

        used_rows, stats = [], {}
        for name, filename in SURVEYS.items():
            survey = read_xtf(self.xtf_dir / filename)
            with np.load(self.table_dir / f"first_bottom_track_{name}.npz") as tracking:
                data, survey_used_rows = prepare_seabed_geometry(survey, name, legs, tracking, utm_epsg)

            data["filename"] = np.asarray(filename)
            np.savez_compressed(self.geometry_path(name), **data)
            used_rows.extend(survey_used_rows)
            valid_port = np.isfinite(data["first_bottom_score_port"])
            valid_stbd = np.isfinite(data["first_bottom_score_stbd"])
            stats[name] = {
                "rows": int(np.count_nonzero(data["in_analysis_leg"])),
                "port": float(np.nanmedian(data["first_bottom_port"])),
                "stbd": float(np.nanmedian(data["first_bottom_stbd"])),
                "low_port": float(np.mean(data["low_score_port"][valid_port])),
                "low_stbd": float(np.mean(data["low_score_stbd"][valid_stbd])),
            }
            del survey, data
            gc.collect()

        pd.DataFrame(used_rows).to_csv(self.table_dir / "geometry_legs_used.csv", index=False)
        return stats

    def build_colocation(self):
        """Find nearest cross-pass anchors and record colocated candidates."""
        config = AnchorConfig()
        anchors, epsg_values = {}, []

        for name in SURVEYS:
            with np.load(self.geometry_path(name)) as geometry:
                anchors[name], epsg = build_anchors(geometry, name, config)
            np.savez_compressed(self.anchor_path(name), **anchor_arrays(anchors[name], epsg, config))
            epsg_values.append(epsg)

        if len(set(epsg_values)) != 1:
            raise RuntimeError("Gli anchor files non usano lo stesso sistema UTM.")

        summary_rows, counts = [], {}
        for a, b in POSITIVE_PAIRS:
            candidates, distances = build_candidate_table(a, b, anchors[a], anchors[b], config)
            candidates.to_csv(self.candidate_path(a, b), index=False)
            summary_rows.append(summarize_positive_pair(a, b, anchors[a], anchors[b], candidates, distances, config))
            counts[f"{a}-{b}"] = len(candidates)

        for a, b in CONTROL_PAIRS:
            summary_rows.append(summarize_control_pair(a, b, anchors[a], anchors[b], config))

        pd.DataFrame(summary_rows).to_csv(self.table_dir / "colocation_summary.csv", index=False)
        del anchors
        gc.collect()
        return counts

    def build_positive_patches(self):
        """Rasterize curated colocated candidates as paired geocoded 50 m patches."""
        config = PositivePatchConfig()
        all_metadata, global_patch_id, counts = [], 0, {}

        for a, b in POSITIVE_PAIRS:
            centres = preselect_centres(pd.read_csv(self.candidate_path(a, b)), config)
            if centres.empty:
                counts[f"{a}-{b}"] = 0
                continue

            survey_a = read_xtf(self.xtf_dir / SURVEYS[a])
            survey_b = read_xtf(self.xtf_dir / SURVEYS[b])
            geom_a, geom_b = self.load_geometry(a), self.load_geometry(b)
            rows = []
            patches_a, patches_b, masks_a, masks_b, counts_a, counts_b = [], [], [], [], [], []

            for row in centres.itertuples(index=False):
                pa, ca, ma = build_mosaic(survey_a, geom_a, row.midpoint_east, row.midpoint_north, config)
                pb, cb, mb = build_mosaic(survey_b, geom_b, row.midpoint_east, row.midpoint_north, config)
                coverage_a, coverage_b = float(ma.mean()), float(mb.mean())
                joint = float((ma & mb).mean())
                if coverage_a < config.min_coverage_each or coverage_b < config.min_coverage_each or joint < config.min_joint_coverage:
                    continue

                global_patch_id += 1
                record = row._asdict()
                record.update(
                    patch_id=global_patch_id, pair=f"{a}-{b}", coverage_a=coverage_a, coverage_b=coverage_b,
                    joint_coverage=joint, patch_size_m=config.patch_size_m,
                    grid_resolution_m=config.grid_resolution_m, representation="geocoded_log1p_mean",
                )
                rows.append(record)
                patches_a.append(pa); patches_b.append(pb)
                masks_a.append(ma); masks_b.append(mb)
                counts_a.append(ca); counts_b.append(cb)

            metadata = pd.DataFrame(rows)
            counts[f"{a}-{b}"] = len(metadata)
            if metadata.empty:
                del survey_a, survey_b, geom_a, geom_b
                gc.collect()
                continue

            patches_a, patches_b = np.stack(patches_a).astype(np.float32), np.stack(patches_b).astype(np.float32)
            masks_a, masks_b = np.stack(masks_a).astype(bool), np.stack(masks_b).astype(bool)
            counts_a, counts_b = np.stack(counts_a).astype(np.uint32), np.stack(counts_b).astype(np.uint32)

            out = self.patch_dir / f"positive_pairs_{a}_{b}.npz"
            np.savez_compressed(
                out,
                patch_id=metadata["patch_id"].to_numpy(np.int64),
                match_id=metadata["match_id"].to_numpy(np.int64),
                center_east=metadata["midpoint_east"].to_numpy(float),
                center_north=metadata["midpoint_north"].to_numpy(float),
                patch_a=patches_a, patch_b=patches_b,
                mask_a=masks_a, mask_b=masks_b,
                count_a=counts_a, count_b=counts_b,
                coverage_a=metadata["coverage_a"].to_numpy(float),
                coverage_b=metadata["coverage_b"].to_numpy(float),
                joint_coverage=metadata["joint_coverage"].to_numpy(float),
                heading_difference_deg=metadata["heading_difference_deg"].to_numpy(float),
                look_difference_deg=metadata["look_difference_deg"].to_numpy(float),
                incidence_difference_deg=metadata["incidence_difference_deg"].to_numpy(float),
                view_class=metadata["view_class"].to_numpy(str),
                patch_size_m=np.float64(config.patch_size_m),
                grid_resolution_m=np.float64(config.grid_resolution_m),
                representation=np.asarray("geocoded_log1p_mean"),
            )

            metadata["patch_npz"] = str(out)
            metadata["pair_local_index"] = np.arange(len(metadata), dtype=int)
            all_metadata.append(metadata)
            del survey_a, survey_b, geom_a, geom_b
            del patches_a, patches_b, masks_a, masks_b, counts_a, counts_b
            gc.collect()

        if not all_metadata:
            raise RuntimeError("No positive benchmark patches passed the coverage thresholds.")

        metadata = pd.concat(all_metadata, ignore_index=True)
        metadata.to_csv(self.table_dir / "positive_pair_metadata.csv", index=False)
        summary = metadata.groupby(["pair", "view_class"], dropna=False).agg(
            patches=("patch_id", "count"),
            median_coverage_a=("coverage_a", "median"),
            median_coverage_b=("coverage_b", "median"),
            median_joint_coverage=("joint_coverage", "median"),
            median_heading_difference_deg=("heading_difference_deg", "median"),
            median_look_difference_deg=("look_difference_deg", "median"),
        ).reset_index()
        summary.to_csv(self.table_dir / "positive_pair_summary.csv", index=False)
        return metadata, counts

    def build_null_patches(self):
        """Build within-site and isolated-control negative pairs and patch banks."""
        config = NullPatchConfig()
        positive_meta = pd.read_csv(self.table_dir / "positive_pair_metadata.csv")
        banks = {}

        for name in SURVEYS:
            survey = read_xtf(self.xtf_dir / SURVEYS[name])
            metadata, arrays = build_patch_bank(survey, self.load_geometry(name), self.load_anchors(name), name, config)
            np.savez_compressed(self.patch_dir / f"patch_bank_{name}.npz", **arrays)
            banks[name] = metadata
            del survey, arrays
            gc.collect()

        bank_metadata_path = self.table_dir / "patch_bank_metadata.csv"
        bank_metadata = pd.concat(banks.values(), ignore_index=True)
        bank_metadata.to_csv(bank_metadata_path, index=False)

        bank_metadata = pd.read_csv(bank_metadata_path)
        banks = {
            name: bank_metadata.loc[bank_metadata["survey"] == name].copy().reset_index(drop=True)
            for name in SURVEYS
        }

        all_null, counts = [], {}
        for a, b in POSITIVE_PAIRS:
            positives = positive_meta.loc[positive_meta["pair"] == f"{a}-{b}"].copy()
            negatives = select_within_site_negatives(a, b, banks[a], banks[b], positives, config)
            all_null.append(negatives)
            counts[f"{a}-{b}"] = len(negatives)

        for target in ("pass_a", "pass_b", "pass_c"):
            n_pairs = min(config.control_pairs_per_survey, len(banks["control"]) * len(banks[target]))
            negatives = select_control_negatives(target, banks["control"], banks[target], n_pairs)
            all_null.append(negatives)
            counts[f"control-{target}"] = len(negatives)

        metadata = pd.concat(all_null, ignore_index=True)
        metadata.insert(0, "null_pair_id", np.arange(1, len(metadata) + 1, dtype=np.int64))
        metadata.to_csv(self.table_dir / "null_pair_metadata.csv", index=False)

        summary = metadata.groupby(["null_type", "survey_a", "survey_b"], dropna=False).agg(
            pairs=("null_pair_id", "count"),
            median_center_separation_m=("center_separation_m", "median"),
            median_heading_difference_deg=("heading_difference_deg", "median"),
            median_look_difference_deg=("look_difference_deg", "median"),
            median_incidence_difference_deg=("incidence_difference_deg", "median"),
        ).reset_index()
        summary.to_csv(self.table_dir / "null_pair_summary.csv", index=False)
        return metadata, counts

    def write_manifest(self, legs_table, legs, first_bottom, geometry, colocated,
                       positives, positive_counts, nulls, null_counts):
        """Describe the reproducible dataset schema, counts and generated artifacts."""
        patch_config = PositivePatchConfig()
        utm_epsg = int(legs_table["utm_epsg"].dropna().iloc[0])

        positive_views = (
            positives.groupby(["pair", "view_class"])
            .size()
            .rename("count")
            .reset_index()
        )

        manifest = {
            "schema_version": 1,
            "generator": "scripts/build_dataset.py",
            "surveys": SURVEYS,
            "coordinate_system": {"utm_epsg": utm_epsg},
            "patches": {
                "representation": "geocoded_log1p_mean",
                "patch_size_m": patch_config.patch_size_m,
                "grid_resolution_m": patch_config.grid_resolution_m,
            },
            "counts": {
                "survey_legs": {k: int(v) for k, v in legs.items()},
                "first_bottom_blocks": {k: int(v["blocks"]) for k, v in first_bottom.items()},
                "analysis_rows": {k: int(v["rows"]) for k, v in geometry.items()},
                "colocation_candidates": {k: int(v) for k, v in colocated.items()},
                "positive_pairs": {
                    "total": int(len(positives)),
                    "by_pair": {k: int(v) for k, v in positive_counts.items()},
                    "by_pair_and_view": {
                        f"{row.pair}/{row.view_class}": int(row.count)
                        for row in positive_views.itertuples(index=False)
                    },
                },
                "null_pairs": {
                    "total": int(len(nulls)),
                    "by_pair": {k: int(v) for k, v in null_counts.items()},
                    "by_type": {k: int(v) for k, v in nulls["null_type"].value_counts().items()},
                },
            },
            "artifacts": {
                "tables": sorted(
                    path.name
                    for path in self.table_dir.iterdir()
                    if path.suffix in {".csv", ".npz"}
                ),
                "patches": sorted(path.name for path in self.patch_dir.glob("*.npz")),
            },
        }

        path = self.project_root / "outputs" / "manifest.json"
        with open(path, "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, sort_keys=True)
            file.write("\n")
        return path

    def build_all(self):
        """Run dataset-building stages in dependency order and write the manifest."""
        legs_table, legs = self.build_legs()
        print("[1/6] legs:", ", ".join(f"{k}={v}" for k, v in legs.items()))

        first_bottom = self.build_first_bottom()
        print("[2/6] first-bottom:", ", ".join(f"{k}={v['blocks']}" for k, v in first_bottom.items()))

        geometry = self.build_geometry()
        print("[3/6] geometry:", ", ".join(f"{k}={v['rows']}" for k, v in geometry.items()))

        colocated = self.build_colocation()
        print("[4/6] candidates:", ", ".join(f"{k}={v}" for k, v in colocated.items()))

        positives, positive_counts = self.build_positive_patches()
        print("[5/6] positives:", ", ".join(f"{k}={v}" for k, v in positive_counts.items()))

        nulls, null_counts = self.build_null_patches()
        print("[6/6] null:", ", ".join(f"{k}={v}" for k, v in null_counts.items()))

        manifest = self.write_manifest(
            legs_table, legs, first_bottom, geometry, colocated,
            positives, positive_counts, nulls, null_counts,
        )
        print(f"Dataset ready: {len(positives)} positives, {len(nulls)} null pairs.")
        print(f"Manifest: {manifest}")
