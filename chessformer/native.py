"""Read the authors' Lc0 V6/V7 game chunks without inventing search targets.

Format reference: daniel-monroe/lczero-training at 7b3ee3fcdb49d8abeb034718935b099134c11882,
tf/chunkparser.py. V7 here means that repository's 8,396-byte rescored record.
The V6 final float is NOT a short-term reward; ST is derived from a full game.
"""

from dataclasses import dataclass
import gzip
import hashlib
from pathlib import Path
import struct
import tarfile

import numpy as np

from .encoding import NUM_ACTIONS

RECORD_SIZES = {6: 8356, 7: 8396}
INPUT_FORMATS = (1, 2, 3, 4, 5, 132, 133)
NATIVE_PREFIX = "lc0-training-v1:"


def native_encoding(input_format):
    if input_format not in INPUT_FORMATS:
        raise ValueError(f"unsupported Lc0 input format: {input_format}")
    return NATIVE_PREFIX + str(input_format)


def short_term_values(rewards, decay=5 / 6):
    """Authors' future EMA, including current Q and reversing mover each ply.

    The last recorded root Q initializes the tail; no synthetic terminal reward.
    Compute BEFORE downsampling a complete chronological game.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 1 or not len(rewards) or not np.isfinite(rewards).all():
        raise ValueError("short-term targets require a nonempty finite reward sequence")
    if not 0 <= decay < 1 or np.any(np.abs(rewards) > 1.00001):
        raise ValueError("invalid decay or reward range")
    values = np.empty_like(rewards)
    values[-1] = rewards[-1]
    for i in range(len(values) - 2, -1, -1):
        values[i] = (1 - decay) * rewards[i] - decay * values[i + 1]
    return values.astype(np.float32)


def qd_to_wdl(q, d):
    values = np.array([(1 - d + q) / 2, d, (1 - d - q) / 2], dtype=np.float32)
    if not np.isfinite(values).all() or values.min() < -1e-5 or values.max() > 1 + 1e-5:
        raise ValueError(f"invalid Q/D distribution: q={q}, d={d}")
    values = values.clip(0, 1)
    return values / values.sum()


def record_header(raw):
    if len(raw) < 8:
        raise ValueError("truncated Lc0 header")
    version, input_format = struct.unpack_from("<II", raw)
    if version not in RECORD_SIZES or len(raw) != RECORD_SIZES[version]:
        raise ValueError(f"unsupported/truncated Lc0 record: version {version}, {len(raw)} bytes")
    native_encoding(input_format)
    return version, input_format


def decode_inputs(raw, input_format):
    inputs = np.zeros((112, 64), dtype=np.float32)
    # Writer reverses bits inside each byte before storing its uint64 planes.
    # np.unpackbits(big) therefore gives a1,b1,...,h8 in that exact order.
    inputs[:104] = np.unpackbits(np.frombuffer(raw, dtype=np.uint8, count=832, offset=7440)).reshape(104, 64)
    us_q, us_k, them_q, them_k, side_ep, clock, flags = raw[8272:8279]
    if input_format == 1:
        if any(flag not in (0, 1) for flag in (us_q, us_k, them_q, them_k, side_ep)):
            raise ValueError("classical input flags must be 0 or 1")
        inputs[104:109] = np.array([us_q, us_k, them_q, them_k, side_ep])[:, None]
    else:
        for channel, row, byte in ((104, 0, us_q), (104, 7, them_q), (105, 0, us_k), (105, 7, them_k)):
            inputs[channel, row * 8:row * 8 + 8] = [(byte >> f) & 1 for f in range(8)]
        if input_format == 2:
            if side_ep not in (0, 1):
                raise ValueError("side-to-move flag must be 0 or 1")
            inputs[108] = side_ep
        else:
            inputs[108, 56:64] = [(side_ep >> f) & 1 for f in range(8)]
    # This normalization matches the pinned TRAINER, including its legacy /99.
    inputs[109] = clock / (100 if input_format > 3 else 99)
    inputs[110] = int(input_format in (132, 133) and bool(flags & 128))
    inputs[111] = 1
    return inputs.T.copy()


def decode_record(raw, short_term=None):
    version, input_format = record_header(raw)
    policy = np.frombuffer(raw, dtype="<f4", count=NUM_ACTIONS, offset=8).copy()
    if not np.isfinite(policy).all() or np.any((policy < 0) & (policy != -1)):
        raise ValueError("policy must contain finite nonnegative probabilities or illegal sentinel -1")
    legal = policy >= 0
    policy[~legal] = 0
    mass = policy.sum(dtype=np.float64)
    if mass <= 0 or not np.isclose(mass, 1, atol=1e-3):
        raise ValueError(f"invalid search-policy mass: {mass}")
    policy /= mass
    root_q = struct.unpack_from("<f", raw, 8280)[0]
    root_d = struct.unpack_from("<f", raw, 8288)[0]
    qd_to_wdl(root_q, root_d)
    result_q, result_d = struct.unpack_from("<ff", raw, 8308)
    wdl = qd_to_wdl(result_q, result_d)
    if short_term is None:
        if version != 7:
            raise ValueError("V6 short-term reward requires the complete chronological game")
        short_term, short_draw = struct.unpack_from("<ff", raw, 8352)
        qd_to_wdl(short_term, short_draw)
    if not np.isfinite(short_term) or abs(short_term) > 1.00001:
        raise ValueError("invalid short-term reward")
    return {"inputs": decode_inputs(raw, input_format), "legal": legal, "policy": policy, "wdl": wdl,
            "q": np.float32(np.clip(root_q, -1, 1)), "st": np.float32(np.clip(short_term, -1, 1)),
            "q_valid": np.bool_(True), "st_valid": np.bool_(True)}


@dataclass
class NativeGame:
    source: str
    records: list[bytes]
    input_format: int
    version: int
    game_id: str
    content_sha256: str
    short_term: np.ndarray

    def samples(self):
        for ply, raw in enumerate(self.records):
            # A deleted frame still contributes its chronological Q to the EMA.
            if raw[8278] & 64:
                continue
            yield ply, decode_record(raw, self.short_term[ply])


def read_game(stream, source="<stream>", *, max_plies=8192, expected_input_format=None):
    """Read at most one bounded game; malformed/truncated chunks fail closed."""
    from .native_validation import ChronologyValidator
    if max_plies < 1:
        raise ValueError("max_plies must be positive")
    header = stream.read(8)
    if len(header) != 8:
        raise ValueError(f"{source}: empty/truncated game")
    version, input_format = struct.unpack("<II", header)
    if version not in RECORD_SIZES:
        raise ValueError(f"{source}: unsupported record version {version}")
    native_encoding(input_format)
    if expected_input_format is not None and expected_input_format != input_format:
        raise ValueError(f"{source}: expected input format {expected_input_format}, got {input_format}")
    records, qs = [], []
    content_hash, identity = hashlib.sha256(), hashlib.sha256()
    previous_side = None
    chronology = ChronologyValidator()
    while header:
        if len(records) >= max_plies:
            raise ValueError(f"{source}: exceeds max_plies={max_plies}")
        raw = header + stream.read(RECORD_SIZES[version] - 8)
        if record_header(raw) != (version, input_format):
            raise ValueError(f"{source}: mixed versions/input formats in a game")
        records.append(raw)
        content_hash.update(raw)
        side = bool(raw[8278] & 128) if input_format >= 3 else bool(raw[8276])
        if previous_side is not None and side == previous_side:
            raise ValueError(f"{source}: non-alternating mover; EMA requires unsampled chronological games")
        previous_side = side
        try:
            chronology.accept(raw)
        except ValueError as error:
            raise ValueError(f"{source}: frame {len(records) - 1}: {error}") from error
        # Game identity excludes rewards, policies and rescoring flags, so relabeling
        # identical positions cannot send the same game to two different splits.
        identity.update(struct.pack("<I", input_format))
        identity.update(raw[7440:7536])  # current 12 piece planes only
        identity.update(raw[8272:8278])  # rights, EP/side, rule50
        identity.update(bytes([raw[8278] & 135]))  # transform + mover, not adjudication
        root_q, root_d = struct.unpack_from("<f", raw, 8280)[0], struct.unpack_from("<f", raw, 8288)[0]
        qd_to_wdl(root_q, root_d)
        qs.append(root_q)
        header = stream.read(8)
    if version == 6:
        st = short_term_values(qs)
    else:
        st = np.array([struct.unpack_from("<f", raw, 8352)[0] for raw in records], dtype=np.float32)
        for raw in records:
            qd_to_wdl(*struct.unpack_from("<ff", raw, 8352))
    return NativeGame(source, records, input_format, version, identity.hexdigest(), content_hash.hexdigest(), st)


def iter_games(paths, *, max_plies=8192, expected_input_format=None):
    """Stream .gz games or tar/tar.gz archives without extracting to the filesystem."""
    options = dict(max_plies=max_plies, expected_input_format=expected_input_format)
    for path in map(Path, paths):
        if path.name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2")):
            with tarfile.open(path, "r|*") as archive:
                for member in archive:
                    if not member.isfile() or not member.name.endswith(".gz"):
                        continue
                    with archive.extractfile(member) as compressed:
                        with gzip.GzipFile(fileobj=compressed) as stream:
                            yield read_game(stream, f"{path}:{member.name}", **options)
        else:
            with gzip.open(path, "rb") as stream:
                yield read_game(stream, str(path), **options)
