# write a test that checks a transcription against a reference sentence
from tmh.transcribe import transcribe_from_audio_path

file_path = "/data/asr/asr/slt/wav/t2un3016.wv1.wav"
transcription = "Det visste i varje fall näsan."

asr_transcription = transcribe_from_audio_path(file_path)

assert transcribe_from_audio_path(file_path) == transcription, "Should be True"