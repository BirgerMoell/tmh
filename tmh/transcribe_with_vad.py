# from vad import extract_speak_segments
import torchaudio
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import json

from pyannote.audio.pipelines import VoiceActivityDetection
# from language_files import get_model

pipeline = VoiceActivityDetection(segmentation="pyannote/segmentation")


language_dict = {
    "Swedish": "KBLab/wav2vec2-large-voxrex-swedish",
    "English": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
    "Russian": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "Spanish": "facebook/wav2vec2-large-xlsr-53-spanish",
    "French": "facebook/wav2vec2-large-xlsr-53-french",
    "Persian": "m3hrdadfi/wav2vec2-large-xlsr-persian",
    "Dutch": "facebook/wav2vec2-large-xlsr-53-dutch",
    "Portugese": "facebook/wav2vec2-large-xlsr-53-portuguese",
    "Chinese": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "German": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
    "Greek": "lighteternal/wav2vec2-large-xlsr-53-greek",
    "Hindi": "theainerd/Wav2Vec2-large-xlsr-hindi",
    "Italian": "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
    "Turkish": "cahya/wav2vec2-base-turkish-artificial-cv",
    "Vietnamese": "leduytan93/Fine-Tune-XLSR-Wav2Vec2-Speech2Text-Vietnamese",
    "Catalan": "ccoreilly/wav2vec2-large-100k-voxpopuli-catala",
    "Japanese": "vumichien/wav2vec2-large-xlsr-japanese-hiragana",
    "Tamil": "vumichien/wav2vec2-large-xlsr-japanese-hiragana",
    "Indonesian": "indonesian-nlp/wav2vec2-large-xlsr-indonesian",
    "Dhivevi": "shahukareem/wav2vec2-large-xlsr-53-dhivehi",
    "Polish": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "Thai": "sakares/wav2vec2-large-xlsr-thai-demo",
    "Hebrew": "imvladikon/wav2vec2-large-xlsr-53-hebrew",
    "Mongolian": "sammy786/wav2vec2-large-xlsr-mongolian",
    "Czech": "arampacha/wav2vec2-large-xlsr-czech",
    "Icelandic": "m3hrdadfi/wav2vec2-large-xlsr-icelandic",
    "Irish": "jimregan/wav2vec2-large-xlsr-irish-basic",
    "Kinyarwanda": "lucio/wav2vec2-large-xlsr-kinyarwanda",
    "Lithuanian": "DeividasM/wav2vec2-large-xlsr-53-lithuanian",
    "Hungarian": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "Finnish": "aapot/wav2vec2-large-xlsr-53-finnish",
    "Arabic": "asafaya/bert-base-arabic"
    }


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
    # print("extracting speaker segments")
    # print(vad)
    return(vad.for_json())

def change_sample_rate(audio_path, new_sample_rate=16000):
    audio_to_resample, sr = librosa.load(audio_path)
    resampled_audio = librosa.resample(audio_to_resample, sr, new_sample_rate)
    resampled_tensor = torch.tensor([resampled_audio])
    return resampled_tensor


def transcribe_from_audio_path_split_on_speech(audio_path, language="Swedish", model="", save_to_file="", output_format="json"):
    """
    Creates a transcription of an audio file, and outputs the
    result in one of the formats json (which is the default),
    or srt.

    If the 'save_to_file' parameter is set to a file name, the
    results will also be written to file. 

    The srt format can be used to create accurately timed subtitles 
    for videos. If 'audio_path' is the audio track of a video file, 
    a video player like VLC will create those subtitles automatically, 
    given a file in the srt format.
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        ## change sample rate to 16000
        waveform = change_sample_rate(audio_path)
        sample_rate = 16000
    segments = extract_speak_segments(audio_path)
    transcriptions = []

    try:
        model_id = language_dict[language]
    except KeyError:
        print("No language model found for %s." %language)

    if model:
        model_id = model

    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)

    for segment in segments['content']:
        x = waveform[:,int(segment['segment']['start']*sample_rate): int(segment['segment']['end']*sample_rate)]
        with torch.no_grad():
            logits = model(x).logits
        pred_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(pred_ids)
        full_transcript = {   
            "transcription": transcription[0].encode('utf8').decode(),
            "start": segment['segment']['start'],
            "end": segment['segment']['end']
        }
        transcriptions.append(full_transcript)

    d = {}
    d['transcripts'] = transcriptions
    result = ""
    if output_format == 'json' :
        result = json.dumps(d,
                            sort_keys=False,
                            indent=4,
                            ensure_ascii=False).encode('utf8').decode()
    elif output_format == 'srt' :
        subtitle_id = 0
        for item in transcriptions :
            transcription = item['transcription']
            start = item['start']
            end = item['end']
            result += str(subtitle_id)
            result += '\n'
            result += time_format(start)
            result += ' --> '
            result += time_format(end)
            result += '\n'
            result += transcription
            result += '\n\n'
            subtitle_id += 1
    if save_to_file :
        f = open(save_to_file, "w")
        f.write( result )
        f.close()
    return result

                       
def time_format( t ) :
    """
    Produces the time format expected by the srt format,
    given a parameter t representing a number of seconds.
    """
    hours = int(t/3600)
    minutes = int((t-hours*3600)/60)
    seconds = int(t-hours*3600-minutes*60)
    fraction = int((t%1)*100)
    return str(hours) + ":" + str(minutes) + ":" + str(seconds) + "." + str(fraction)
