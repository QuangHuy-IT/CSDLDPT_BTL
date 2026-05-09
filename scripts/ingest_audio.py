import argparse
import re
from pathlib import Path

from database.db import get_conn
from features.extractor import extract_features


NOTE_TOKEN_RE = re.compile(r"^(?P<note>[A-G])(?P<accidentals>s|#)?(?P<octave>-?\d{1,2})$")


def parse_note_from_filename(stem):
    for token in stem.split("_"):
        match = NOTE_TOKEN_RE.match(token)
        if not match:
            continue
        note = match.group("note")
        accidental = match.group("accidentals")
        octave = match.group("octave")
        if accidental in {"s", "#"}:
            note = f"{note}#"
        return note, int(octave)
    return None, None


def parse_dynamic_from_filename(stem):
    stem = stem.lower()
    if "fortissimo" in stem:
        return "ff"
    if "pianissimo" in stem:
        return "pp"
    if "mezzo-forte" in stem:
        return "mf"
    if "mezzo-piano" in stem:
        return "mp"
    if "forte" in stem:
        return "f"
    if "piano" in stem:
        return "p"
    return None


def parse_duration_from_filename(stem):
    tokens = stem.lower().split("_")
    if "phrase" in tokens:
        return "phrase"
    if "very-long" in tokens:
        return "very-long"
    if "long" in tokens:
        return "long"

    duration_map = {
        "025": "0.25",
        "05": "0.5",
        "1": "1",
        "15": "1.5",
    }
    for token in tokens:
        if token in duration_map:
            return duration_map[token]
    return None


def parse_playing_technique_from_filename(stem):
    tokens = stem.split("_")
    if not tokens:
        return None
    technique = tokens[-1].strip()
    if not technique:
        return None

    lower = technique.lower()
    if NOTE_TOKEN_RE.match(technique):
        return None
    duration_tokens = {"025", "05", "1", "15", "very-long", "long", "phrase"}
    dynamic_tokens = {
        "piano",
        "forte",
        "fortissimo",
        "pianissimo",
        "mezzo-forte",
        "mezzo-piano",
        "crescendo",
        "decrescendo",
    }

    if lower in duration_tokens or lower in dynamic_tokens:
        return None

    if len(tokens) >= 2:
        prev = tokens[-2].strip()
        prev_lower = prev.lower() if prev else ""
        if prev and not NOTE_TOKEN_RE.match(prev) and prev_lower not in duration_tokens:
            return f"{prev}_{technique}"

    return technique


def upsert_audio_file(cur, row):
    cur.execute(
        """
        INSERT INTO audiofiles (
            note,
            octave,
            dynamic,
            duration_file,
            duration_note,
            name_intrument,
            playing_technique,
            file_path
        )
        VALUES (%(note)s, %(octave)s, %(dynamic)s, %(duration_file)s,
                %(duration_note)s, %(name_intrument)s, %(playing_technique)s,
                %(file_path)s)
        ON CONFLICT (file_path) DO UPDATE SET
            note = EXCLUDED.note,
            octave = EXCLUDED.octave,
            dynamic = EXCLUDED.dynamic,
            duration_file = EXCLUDED.duration_file,
            duration_note = EXCLUDED.duration_note,
            name_intrument = EXCLUDED.name_intrument,
            playing_technique = EXCLUDED.playing_technique
        RETURNING id
        """,
        row,
    )
    return cur.fetchone()[0]


def upsert_features(cur, row):
    cur.execute(
        """
        INSERT INTO features (
            audio_file_id,
            mean_energy,
            mfcc,
            zcr,
            silence_ratio,
            decay_rate,
            harmonicity_ratio,
            f0_mean,
            spectral_centroid,
            bandwidth,
            spectral_flux,
            spectral_roughness,
            feature_vector
        )
        VALUES (
            %(audio_file_id)s,
            %(mean_energy)s,
            %(mfcc)s,
            %(zcr)s,
            %(silence_ratio)s,
            %(decay_rate)s,
            %(harmonicity_ratio)s,
            %(f0_mean)s,
            %(spectral_centroid)s,
            %(bandwidth)s,
            %(spectral_flux)s,
            %(spectral_roughness)s,
            %(feature_vector)s
        )
        ON CONFLICT (audio_file_id) DO UPDATE SET
            mean_energy = EXCLUDED.mean_energy,
            mfcc = EXCLUDED.mfcc,
            zcr = EXCLUDED.zcr,
            silence_ratio = EXCLUDED.silence_ratio,
            decay_rate = EXCLUDED.decay_rate,
            harmonicity_ratio = EXCLUDED.harmonicity_ratio,
            f0_mean = EXCLUDED.f0_mean,
            spectral_centroid = EXCLUDED.spectral_centroid,
            bandwidth = EXCLUDED.bandwidth,
            spectral_flux = EXCLUDED.spectral_flux,
            spectral_roughness = EXCLUDED.spectral_roughness,
            feature_vector = EXCLUDED.feature_vector
        """,
        row,
    )


def main():
    parser = argparse.ArgumentParser(description="Ingest audio files into PostgreSQL.")
    parser.add_argument(
        "--audio-dir",
        default="dataset/string_audio",
        help="Directory with string audio wav files.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files.")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        raise SystemExit(f"Audio dir not found: {audio_dir}")

    audio_exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    audio_files = sorted(
        [
            path
            for path in audio_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in audio_exts
        ]
    )
    if args.limit:
        audio_files = audio_files[: args.limit]

    conn = get_conn()
    count = 0

    with conn:
        with conn.cursor() as cur:
            for audio_path in audio_files:
                instrument_name = audio_path.parent.name

                features = extract_features(str(audio_path))
                if not features:
                    continue

                note, octave = parse_note_from_filename(audio_path.stem)
                dynamic = parse_dynamic_from_filename(audio_path.stem)
                duration_note = parse_duration_from_filename(audio_path.stem) or "full"
                playing_technique = (
                    parse_playing_technique_from_filename(audio_path.stem) or "unknown"
                )

                audio_row = {
                    "note": note,
                    "octave": octave,
                    "dynamic": dynamic,
                    "duration_file": features["duration_sec"],
                    "duration_note": duration_note,
                    "name_intrument": instrument_name,
                    "playing_technique": playing_technique,
                    "file_path": str(audio_path),
                }

                audio_id = upsert_audio_file(cur, audio_row)

                feature_row = {
                    "audio_file_id": audio_id,
                    "mean_energy": features["mean_energy"],
                    "mfcc": features["mfcc"],
                    "zcr": features["zcr"],
                    "silence_ratio": features["silence_ratio"],
                    "decay_rate": features["decay_rate"],
                    "harmonicity_ratio": features["harmonicity_ratio"],
                    "f0_mean": features["f0_mean"],
                    "spectral_centroid": features["spectral_centroid"],
                    "bandwidth": features["bandwidth"],
                    "spectral_flux": features["spectral_flux"],
                    "spectral_roughness": features["spectral_roughness"],
                    "feature_vector": features["feature_vector"],
                }

                upsert_features(cur, feature_row)

                count += 1
                if count % 50 == 0:
                    conn.commit()
                    print(f"Ingested {count} files...")

    conn.commit()
    print(f"Done. Total ingested: {count}")


if __name__ == "__main__":
    main()
