# write a test that checks a transcription against a reference sentence
from tmh.transcribe import TranscribeModel

file_path = "/data/asr/asr/slt/wav/t2un3016.wv1.wav"
transcription = "Det visste i varje fall näsan."

normal = TranscribeModel()
norm_t = normal.transcribe(file_path)

assert norm_t == transcription, "Normal transcription does not match reference"

