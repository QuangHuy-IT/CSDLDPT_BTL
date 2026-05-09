CREATE TABLE IF NOT EXISTS audiofiles (
  id SERIAL PRIMARY KEY,
  note VARCHAR(16),
  octave INT,
  dynamic VARCHAR(256),
  duration_file DOUBLE PRECISION,
  duration_note VARCHAR(16),
  name_intrument VARCHAR(256),
  playing_technique VARCHAR(256),
  file_path TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS features (
  id SERIAL PRIMARY KEY,
  audio_file_id INT UNIQUE REFERENCES audiofiles(id) ON DELETE CASCADE,
  mean_energy DOUBLE PRECISION,
  mfcc DOUBLE PRECISION[],
  zcr DOUBLE PRECISION,
  silence_ratio DOUBLE PRECISION,
  decay_rate DOUBLE PRECISION,
  harmonicity_ratio DOUBLE PRECISION,
  f0_mean DOUBLE PRECISION,
  spectral_centroid DOUBLE PRECISION,
  bandwidth DOUBLE PRECISION,
  spectral_flux DOUBLE PRECISION,
  spectral_roughness DOUBLE PRECISION,
  feature_vector DOUBLE PRECISION[]
);

CREATE INDEX IF NOT EXISTS idx_features_audio_file_id ON features(audio_file_id);
CREATE INDEX IF NOT EXISTS idx_audiofiles_name_intrument ON audiofiles(name_intrument);
