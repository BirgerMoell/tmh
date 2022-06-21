from tmh.transcribe import TranscribeModel
from tmh.utils import ensure_wav

file, converted = ensure_wav("tmh/tests/word1.m4a")

normal = TranscribeModel()
norm_t = normal.transcribe(file)
print("Normal:", norm_t)

vad = TranscribeModel(use_vad=True)
vad_t = vad.transcribe(file, output_format="str")
print("VAD:", vad_t)

lm = TranscribeModel(use_lm=True)
lm_t = lm.transcribe(file)
print("LM:", lm_t)

lm_vad = TranscribeModel(use_lm=True, use_vad=True)
lm_vad_t = lm_vad.transcribe(file, output_format="str")
print("LM+VAD:", lm_vad_t)
