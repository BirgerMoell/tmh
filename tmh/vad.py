from pyannote.audio.pipelines import VoiceActivityDetection

pipeline = VoiceActivityDetection(segmentation="pyannote/segmentation")

HYPER_PARAMETERS = {
  # onset/offset activation thresholds
  "onset": 0.5, "offset": 0.5,
  # remove speech regions shorter than that many seconds.
  "min_duration_on": 0.0,
  # fill non-speech regions shorter than that many seconds.
  "min_duration_off": 0.0
}

def extract_speak_segments(audio_path):
    pipeline.instantiate(HYPER_PARAMETERS)
    vad = pipeline(audio_path)
    print("extracting speaker segments")
    print(vad)
    return(vad.for_json())


