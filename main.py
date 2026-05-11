import uuid
from pathlib import Path

import numpy as np
from flask import Flask, abort, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

from database.db import fetch_all_features, get_conn
from features.extractor import extract_features

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

app = Flask(__name__)


def cosine_similarity(vec_a, vec_b):
	a = np.asarray(vec_a, dtype=np.float32)
	b = np.asarray(vec_b, dtype=np.float32)
	if a.size == 0 or b.size == 0:
		return 0.0
	if a.size != b.size:
		size = min(a.size, b.size)
		a = a[:size]
		b = b[:size]
	denom = float(np.linalg.norm(a) * np.linalg.norm(b))
	if denom == 0.0:
		return 0.0
	return float(np.dot(a, b) / denom)


def align_vectors(query_vec, vectors):
	if not vectors:
		return np.asarray(query_vec, dtype=np.float32), []
	min_len = min(len(query_vec), min(len(v) for v in vectors))
	query_vec = np.asarray(query_vec[:min_len], dtype=np.float32)
	trimmed = [np.asarray(v[:min_len], dtype=np.float32) for v in vectors]
	return query_vec, trimmed


def standardize_params(vectors):
	matrix = np.vstack(vectors)
	means = np.mean(matrix, axis=0)
	stds = np.std(matrix, axis=0)
	stds = np.where(stds == 0.0, 1.0, stds)
	return means, stds


def standardize_vector(vec, means, stds):
	return (vec - means) / stds

def format_query_features(features):
	return {
		"mean_energy": f"{features['mean_energy']:.5f}",
		"zcr": f"{features['zcr']:.5f}",
		"f0_median": f"{features['f0_median']:.2f}",
		"spectral_centroid": f"{features['spectral_centroid']:.2f}",
		"bandwidth": f"{features['bandwidth']:.2f}",
	}


@app.route("/", methods=["GET", "POST"])
def index():
	results = []
	message = None
	query_features = None
	query_audio_url = None
	query_filename = None

	if request.method == "POST":
		audio_file = request.files.get("audio")
		save_path = None
		saved_name = None
		existing_name = (request.form.get("existing_file") or "").strip()
		if audio_file and audio_file.filename:
			ext = Path(audio_file.filename).suffix.lower()
			if ext not in ALLOWED_EXTS:
				message = "Unsupported file type."
			else:
				safe_name = secure_filename(audio_file.filename)
				saved_name = f"{uuid.uuid4().hex}_{safe_name}"
				save_path = UPLOAD_DIR / saved_name
				audio_file.save(save_path)
		elif existing_name:
			target = (UPLOAD_DIR / existing_name).resolve()
			if UPLOAD_DIR.resolve() not in target.parents or not target.exists():
				message = "Please choose an audio file."
			else:
				saved_name = existing_name
				save_path = target
		else:
			message = "Please choose an audio file."

		if save_path and not message:
			features = extract_features(str(save_path))
			if not features:
				message = "Could not extract features from the file."
			else:
				query_audio_url = url_for("serve_upload", filename=saved_name)
				query_filename = saved_name
				query_features = format_query_features(features)

				conn = get_conn()
				try:
					with conn:
						rows = fetch_all_features(conn)
				finally:
					conn.close()

				valid_rows = []
				feature_vectors = []
				for (
					audio_id,
					feature_vector,
					f0_median,
					file_path,
					instrument,
					note,
					octave,
					dynamic,
				) in rows:
					if not feature_vector:
						continue
					valid_rows.append(
						(
							audio_id,
							feature_vector,
							f0_median,
							file_path,
							instrument,
							note,
							octave,
							dynamic,
						)
					)
					feature_vectors.append(feature_vector)

				if not feature_vectors:
					message = "No features available in the database."
				else:
					query_vec, feature_vectors = align_vectors(
						features["feature_vector"], feature_vectors
					)
					if query_vec.size == 0:
						message = "Could not normalize features for comparison."
					else:
						query_timbre = query_vec
						timbre_vectors = feature_vectors
						means, stds = standardize_params(timbre_vectors)
						query_norm = standardize_vector(query_timbre, means, stds)

						for idx, row in enumerate(valid_rows):
							(
								audio_id,
								feature_vector,
								f0_median,
								file_path,
								instrument,
								note,
								octave,
								dynamic,
							) = row
							item_timbre = timbre_vectors[idx]
							item_norm = standardize_vector(item_timbre, means, stds)
							
							timbre_sim = cosine_similarity(query_norm, item_norm)
							similarity = 0.5 * (timbre_sim + 1.0)

							results.append(
								{
									"audio_id": audio_id,
									"audio_url": url_for("serve_audio", audio_id=audio_id),
									"similarity": similarity,
									"instrument": instrument,
									"note": note,
									"octave": octave,
									"dynamic": dynamic,
									"file_path": file_path,
								}
							)

						results.sort(key=lambda item: item["similarity"], reverse=True)
						results = results[:5]

	return render_template(
		"index.html",
		results=results,
		message=message,
		query_features=query_features,
		query_audio_url=query_audio_url,
		query_filename=query_filename,
	)


@app.route("/audio/<int:audio_id>")
def serve_audio(audio_id):
	conn = get_conn()
	try:
		with conn:
			with conn.cursor() as cur:
				cur.execute("SELECT file_path FROM audiofiles WHERE id = %s", (audio_id,))
				row = cur.fetchone()
	finally:
		conn.close()

	if not row:
		abort(404)

	audio_path = Path(row[0]).resolve()
	if not audio_path.exists():
		abort(404)

	return send_file(audio_path, as_attachment=False)


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
	target = (UPLOAD_DIR / filename).resolve()
	if UPLOAD_DIR.resolve() not in target.parents:
		abort(404)
	if not target.exists():
		abort(404)
	return send_file(target, as_attachment=False)


if __name__ == "__main__":
	app.run(debug=True)
