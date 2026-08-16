"""Read GPS samples from a DJI subtitle track without decoding video frames."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
import struct
from typing import BinaryIO


class GpsMetadataError(ValueError):
    """Raised when usable DJI GPS metadata cannot be read from an MP4 file."""


@dataclass(frozen=True)
class GpsSample:
    time_seconds: float
    longitude: float
    latitude: float
    gps_status: int
    distance_home_m: float
    height_m: float
    horizontal_speed_m_s: float
    vertical_speed_m_s: float


@dataclass(frozen=True)
class _Atom:
    kind: bytes
    start: int
    size: int
    header_size: int

    @property
    def payload_start(self) -> int:
        return self.start + self.header_size

    @property
    def end(self) -> int:
        return self.start + self.size


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_GPS_PATTERN = re.compile(
    rf"GPS\s*\(\s*(?P<longitude>{_NUMBER})\s*,\s*"
    rf"(?P<latitude>{_NUMBER})\s*,\s*(?P<status>\d+)\s*\)\s*,\s*"
    rf"D\s*(?P<distance>{_NUMBER})m\s*,\s*"
    rf"H\s*(?P<height>{_NUMBER})m\s*,\s*"
    rf"H\.S\s*(?P<horizontal_speed>{_NUMBER})m/s\s*,\s*"
    rf"V\.S\s*(?P<vertical_speed>{_NUMBER})m/s",
    re.IGNORECASE,
)


def parse_dji_subtitle(text: str, *, time_seconds: float = 0.0) -> GpsSample:
    """Parse one DJI subtitle string containing GPS and flight telemetry."""
    match = _GPS_PATTERN.search(text)
    if match is None:
        raise GpsMetadataError("DJI subtitle does not contain recognized GPS metadata")
    values = match.groupdict()
    return GpsSample(
        time_seconds=float(time_seconds),
        longitude=float(values["longitude"]),
        latitude=float(values["latitude"]),
        gps_status=int(values["status"]),
        distance_home_m=float(values["distance"]),
        height_m=float(values["height"]),
        horizontal_speed_m_s=float(values["horizontal_speed"]),
        vertical_speed_m_s=float(values["vertical_speed"]),
    )


def to_local_metres(samples: list[GpsSample]) -> list[tuple[float, float]]:
    """Convert longitude/latitude to local East/North metres from sample zero."""
    if not samples:
        return []
    earth_radius_m = 6_371_000.0
    longitude_zero = math.radians(samples[0].longitude)
    latitude_zero = math.radians(samples[0].latitude)
    cosine_latitude = math.cos(latitude_zero)
    return [
        (
            earth_radius_m
            * cosine_latitude
            * (math.radians(sample.longitude) - longitude_zero),
            earth_radius_m * (math.radians(sample.latitude) - latitude_zero),
        )
        for sample in samples
    ]


def extract_gps_samples(video_path: str | Path) -> list[GpsSample]:
    """Extract all recognized GPS records from the DJI subtitle MP4 track."""
    path = Path(video_path)
    if not path.is_file():
        raise GpsMetadataError(f"video file does not exist: {path}")

    with path.open("rb") as handle:
        handle.seek(0, 2)
        file_size = handle.tell()
        moov = _find_child(handle, 0, file_size, b"moov")
        if moov is None:
            raise GpsMetadataError("MP4 does not contain a moov atom")
        track = _find_dji_subtitle_track(handle, moov)
        if track is None:
            raise GpsMetadataError("MP4 does not contain a DJI.Subtitle track")

        mdia = _required_child(handle, track, b"mdia")
        mdhd = _required_child(handle, mdia, b"mdhd")
        minf = _required_child(handle, mdia, b"minf")
        stbl = _required_child(handle, minf, b"stbl")
        timescale = _read_timescale(handle, mdhd)
        sample_sizes = _read_sample_sizes(
            handle,
            _required_child(handle, stbl, b"stsz"),
        )
        chunk_atom = _find_child(handle, stbl.payload_start, stbl.end, b"stco")
        if chunk_atom is None:
            chunk_atom = _find_child(handle, stbl.payload_start, stbl.end, b"co64")
        if chunk_atom is None:
            raise GpsMetadataError("subtitle track has no chunk-offset table")
        chunk_offsets = _read_chunk_offsets(handle, chunk_atom)
        chunk_layout = _read_sample_to_chunk(
            handle,
            _required_child(handle, stbl, b"stsc"),
        )
        sample_offsets = _expand_sample_offsets(
            sample_sizes,
            chunk_offsets,
            chunk_layout,
        )
        timestamps = _read_timestamps(
            handle,
            _required_child(handle, stbl, b"stts"),
            timescale,
            len(sample_sizes),
        )

        samples: list[GpsSample] = []
        for offset, size, timestamp in zip(
            sample_offsets,
            sample_sizes,
            timestamps,
            strict=True,
        ):
            handle.seek(offset)
            raw_sample = handle.read(size)
            if len(raw_sample) != size:
                raise GpsMetadataError("subtitle sample extends beyond the MP4 file")
            text = _decode_text_sample(raw_sample)
            try:
                samples.append(parse_dji_subtitle(text, time_seconds=timestamp))
            except GpsMetadataError:
                continue

    if not samples:
        raise GpsMetadataError("DJI subtitle track contains no recognized GPS samples")
    return samples


def _iter_atoms(
    handle: BinaryIO,
    start: int,
    end: int,
):
    position = start
    while position + 8 <= end:
        handle.seek(position)
        header = handle.read(8)
        if len(header) != 8:
            break
        size_32, kind = struct.unpack(">I4s", header)
        header_size = 8
        if size_32 == 1:
            extended = handle.read(8)
            if len(extended) != 8:
                raise GpsMetadataError("truncated extended MP4 atom size")
            size = struct.unpack(">Q", extended)[0]
            header_size = 16
        elif size_32 == 0:
            size = end - position
        else:
            size = size_32
        if size < header_size or position + size > end:
            raise GpsMetadataError(f"invalid MP4 atom {kind!r} at byte {position}")
        yield _Atom(kind=kind, start=position, size=size, header_size=header_size)
        position += size


def _find_child(
    handle: BinaryIO,
    start: int,
    end: int,
    kind: bytes,
) -> _Atom | None:
    return next(
        (atom for atom in _iter_atoms(handle, start, end) if atom.kind == kind),
        None,
    )


def _required_child(handle: BinaryIO, parent: _Atom, kind: bytes) -> _Atom:
    child = _find_child(handle, parent.payload_start, parent.end, kind)
    if child is None:
        raise GpsMetadataError(
            f"MP4 atom {parent.kind!r} does not contain required {kind!r} atom"
        )
    return child


def _read_payload(handle: BinaryIO, atom: _Atom) -> bytes:
    handle.seek(atom.payload_start)
    payload = handle.read(atom.size - atom.header_size)
    if len(payload) != atom.size - atom.header_size:
        raise GpsMetadataError(f"truncated MP4 atom {atom.kind!r}")
    return payload


def _find_dji_subtitle_track(handle: BinaryIO, moov: _Atom) -> _Atom | None:
    for atom in _iter_atoms(handle, moov.payload_start, moov.end):
        if atom.kind != b"trak":
            continue
        mdia = _find_child(handle, atom.payload_start, atom.end, b"mdia")
        if mdia is None:
            continue
        hdlr = _find_child(handle, mdia.payload_start, mdia.end, b"hdlr")
        if hdlr is None:
            continue
        payload = _read_payload(handle, hdlr)
        if len(payload) >= 12 and payload[8:12] == b"text" and b"DJI.Subtitle" in payload:
            return atom
    return None


def _read_timescale(handle: BinaryIO, mdhd: _Atom) -> int:
    payload = _read_payload(handle, mdhd)
    if not payload:
        raise GpsMetadataError("empty mdhd atom")
    offset = 20 if payload[0] == 1 else 12
    if len(payload) < offset + 4:
        raise GpsMetadataError("truncated mdhd atom")
    timescale = struct.unpack_from(">I", payload, offset)[0]
    if timescale == 0:
        raise GpsMetadataError("subtitle track has a zero timescale")
    return timescale


def _read_sample_sizes(handle: BinaryIO, stsz: _Atom) -> list[int]:
    payload = _read_payload(handle, stsz)
    if len(payload) < 12:
        raise GpsMetadataError("truncated stsz atom")
    constant_size, sample_count = struct.unpack_from(">II", payload, 4)
    if constant_size:
        return [constant_size] * sample_count
    expected = 12 + sample_count * 4
    if len(payload) < expected:
        raise GpsMetadataError("truncated stsz sample table")
    return list(struct.unpack_from(f">{sample_count}I", payload, 12))


def _read_chunk_offsets(handle: BinaryIO, atom: _Atom) -> list[int]:
    payload = _read_payload(handle, atom)
    if len(payload) < 8:
        raise GpsMetadataError("truncated chunk-offset atom")
    count = struct.unpack_from(">I", payload, 4)[0]
    value_size = 8 if atom.kind == b"co64" else 4
    if len(payload) < 8 + count * value_size:
        raise GpsMetadataError("truncated chunk-offset table")
    code = "Q" if value_size == 8 else "I"
    return list(struct.unpack_from(f">{count}{code}", payload, 8))


def _read_sample_to_chunk(
    handle: BinaryIO,
    stsc: _Atom,
) -> list[tuple[int, int]]:
    payload = _read_payload(handle, stsc)
    if len(payload) < 8:
        raise GpsMetadataError("truncated stsc atom")
    count = struct.unpack_from(">I", payload, 4)[0]
    if len(payload) < 8 + count * 12:
        raise GpsMetadataError("truncated stsc table")
    return [
        struct.unpack_from(">III", payload, 8 + index * 12)[:2]
        for index in range(count)
    ]


def _expand_sample_offsets(
    sample_sizes: list[int],
    chunk_offsets: list[int],
    layout: list[tuple[int, int]],
) -> list[int]:
    if not layout or layout[0][0] != 1:
        raise GpsMetadataError("invalid sample-to-chunk table")
    offsets: list[int] = []
    sample_index = 0
    layout_index = 0
    for chunk_number, chunk_offset in enumerate(chunk_offsets, start=1):
        while (
            layout_index + 1 < len(layout)
            and chunk_number >= layout[layout_index + 1][0]
        ):
            layout_index += 1
        samples_in_chunk = layout[layout_index][1]
        offset = chunk_offset
        for _ in range(samples_in_chunk):
            if sample_index >= len(sample_sizes):
                break
            offsets.append(offset)
            offset += sample_sizes[sample_index]
            sample_index += 1
    if len(offsets) != len(sample_sizes):
        raise GpsMetadataError("chunk table does not cover every subtitle sample")
    return offsets


def _read_timestamps(
    handle: BinaryIO,
    stts: _Atom,
    timescale: int,
    sample_count: int,
) -> list[float]:
    payload = _read_payload(handle, stts)
    if len(payload) < 8:
        raise GpsMetadataError("truncated stts atom")
    entry_count = struct.unpack_from(">I", payload, 4)[0]
    if len(payload) < 8 + entry_count * 8:
        raise GpsMetadataError("truncated stts table")
    timestamps: list[float] = []
    current_time = 0
    for index in range(entry_count):
        count, delta = struct.unpack_from(">II", payload, 8 + index * 8)
        for _ in range(count):
            if len(timestamps) >= sample_count:
                break
            timestamps.append(current_time / timescale)
            current_time += delta
    if len(timestamps) != sample_count:
        raise GpsMetadataError("timing table does not cover every subtitle sample")
    return timestamps


def _decode_text_sample(raw_sample: bytes) -> str:
    if len(raw_sample) < 2:
        return ""
    text_length = struct.unpack_from(">H", raw_sample)[0]
    if text_length <= len(raw_sample) - 2:
        text_bytes = raw_sample[2 : 2 + text_length]
    else:
        text_bytes = raw_sample[2:]
    return text_bytes.rstrip(b"\0").decode("utf-8", errors="replace")
