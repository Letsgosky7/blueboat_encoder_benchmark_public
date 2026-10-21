"""Read a single BlueBoat XTF acquisition into channel arrays and ping metadata.

Input: local .xtf path passed to read_xtf; requires pyxtf.
Output: XTFSurvey with synchronized port/starboard arrays, timestamps and ranges.
Usage: import read_xtf from this module; it is not a CLI."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyxtf


@dataclass
class XTFSurvey:
    """Single side-scan sonar acquisition read from an XTF file."""

    path: Path
    metadata: pd.DataFrame
    port: np.ndarray
    starboard: np.ndarray
    slant_range_port: float
    slant_range_starboard: float


def _build_timestamp(ping) -> datetime:
    """Decode the XTF ping date/time fields into a Python datetime."""
    return datetime(
        year=int(ping.Year), month=int(ping.Month), day=int(ping.Day),
        hour=int(ping.Hour), minute=int(ping.Minute), second=int(ping.Second),
        microsecond=int(ping.HSeconds) * 10_000,
    )


def _channel_name(info) -> str:
    """Normalize an XTF channel descriptor to a readable channel label."""
    return info.ChannelName.decode("utf-8", errors="ignore").strip("\x00").lower()


def read_xtf(path: str | Path) -> XTFSurvey:
    """Read XTF metadata, PORT/STARBOARD matrices and channel slant ranges."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"XTF file not found: {path}")

    file_header, packets = pyxtf.xtf_read(str(path))
    sonar_packets = packets.get(pyxtf.XTFHeaderType.sonar, [])
    if not sonar_packets:
        raise RuntimeError(f"No SONAR pings found in {path.name}")
    if file_header.NumberOfSonarChannels < 2 or len(file_header.sonar_info) < 2:
        raise RuntimeError("File does not contain two valid side-scan SONAR channels.")

    channel_0, channel_1 = map(_channel_name, file_header.sonar_info[:2])
    if channel_0 != "port":
        raise RuntimeError(f"Unexpected channel 0: {channel_0}")
    if channel_1 != "starboard":
        raise RuntimeError(f"Unexpected channel 1: {channel_1}")

    metadata_rows, port_rows, starboard_rows = [], [], []
    slant_range_port = slant_range_starboard = None

    for ping_index, ping in enumerate(sonar_packets):
        if len(ping.data) < 2:
            continue

        port_samples = np.asarray(ping.data[0], dtype=np.uint16)
        starboard_samples = np.asarray(ping.data[1], dtype=np.uint16)
        if port_samples.size == 0 or starboard_samples.size == 0:
            continue

        port_header, starboard_header = ping.ping_chan_headers[:2]
        port_rows.append(port_samples)
        starboard_rows.append(starboard_samples)

        if slant_range_port is None:
            slant_range_port = float(port_header.SlantRange)
        if slant_range_starboard is None:
            slant_range_starboard = float(starboard_header.SlantRange)

        metadata_rows.append({
            "ping_index": ping_index,
            "ping_number": int(ping.PingNumber),
            "timestamp": _build_timestamp(ping),
            "x": float(ping.SensorXcoordinate),
            "y": float(ping.SensorYcoordinate),
            "heading_raw": float(ping.SensorHeading),
            "altitude": float(ping.SensorPrimaryAltitude),
            "sound_velocity": float(ping.SoundVelocity),
            "port_slant_range": float(port_header.SlantRange),
            "starboard_slant_range": float(starboard_header.SlantRange),
            "port_num_samples": int(port_header.NumSamples),
            "starboard_num_samples": int(starboard_header.NumSamples),
        })

    if not port_rows or not starboard_rows:
        raise RuntimeError("No valid SONAR data found.")

    min_samples = min(min(row.size for row in port_rows), min(row.size for row in starboard_rows))
    port = np.stack([row[:min_samples] for row in port_rows])
    starboard = np.stack([row[:min_samples] for row in starboard_rows])

    return XTFSurvey(
        path=path,
        metadata=pd.DataFrame(metadata_rows),
        port=port,
        starboard=starboard,
        slant_range_port=slant_range_port,
        slant_range_starboard=slant_range_starboard,
    )
