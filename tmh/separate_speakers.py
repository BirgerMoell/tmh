# from vad import extract_speak_segments
from librosa.core.spectrum import power_to_db
import torch
from .overlap import overlap_detection
from .transcribe import extract_speaker_embedding
import numpy as np

from sklearn.preprocessing import StandardScaler

from pydub import AudioSegment
from .audio_embeddings import get_audio_embeddings

from pyannote.audio import Inference
from pyannote.audio.pipelines import VoiceActivityDetection
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, MiniBatchKMeans
from pyannote.core import Segment, Timeline
import torchaudio
from speechbrain.pretrained import EncoderClassifier
# from language_files import get_model

pipeline = VoiceActivityDetection(segmentation="pyannote/segmentation")

# speaker embedding
inference = Inference("pyannote/embedding", window="whole")

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


def classify_speakers_based_on_embeddings(X):
    X = StandardScaler().fit_transform(X)
    
    estimator=GaussianMixture(n_components=2)
    y = estimator.fit_predict(X)
    return y


def extract_speaker_embedding(signal, audio):

    audio_file = torch.tensor(audio[signal.start*1000 : signal.end * 1000].get_array_of_samples())
    #embeddings = classifier.encode_batch(audio_file)
    embeddings = inference( {"waveform": audio_file.unsqueeze(0).float(), "sample_rate": 44100} )

    # print(embeddings)
    return embeddings
    


def create_speaker_files_from_audio_path(audio_path):

    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

    # apply pretrained pipeline
    audio = AudioSegment.from_wav(audio_path)

    diarization = pipeline(audio_path)

    speaker1= AudioSegment.empty()
    speaker2 = AudioSegment.empty()

    # print the result
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
        current_audio_segment = audio[turn.start*1000:turn.end*1000]
        if speaker == "SPEAKER_00":
            print("speaker1", speaker)
            speaker1+=current_audio_segment
        elif speaker == "SPEAKER_01":
            print("speaker2", speaker2)
            speaker2+=current_audio_segment

        speaker1.export("speaker1.wav", format="wav")
        speaker2.export("speaker2.wav", format="wav")


def wavlm_speaker_diarization(audio_path):
    from transformers import Wav2Vec2FeatureExtractor, UniSpeechSatForAudioFrameClassification
    import torch

    audio_file = torchaudio.load(audio_path)

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-sd')
    model = UniSpeechSatForAudioFrameClassification.from_pretrained('microsoft/wavlm-base-sd')



    # audio file is decoded on the fly
    inputs = feature_extractor(audio_file, return_tensors="pt")
    logits = model(**inputs).logits
    probabilities = torch.sigmoid(logits[0])

    # labels is a one-hot array of shape (num_frames, num_speakers)
    labels = (probabilities > 0.5).long()
    print(labels)


def create_speaker_files_from_audio_path_old(audio_path):
    """
    Creates speaker files from the audio file at audio_path by splitting based on speaker.

    If the 'save_to_file' parameter is set to a file name, the
    results will also be written to file. 

    The srt format can be used to create accurately timed subtitles 
    for videos. If 'audio_path' is the audio track of a video file, 
    a video player like VLC will create those subtitles automatically, 
    given a file in the srt format.
    """
    
    audio = AudioSegment.from_wav(audio_path)

    if audio.frame_rate != 16000:
        ## change sample rate to 16000
        waveform = audio.set_frame_rate(16_000)

    
    # first we get all the overlaps for each speaker in the file
    overlaps = overlap_detection(audio_path)
    print(overlaps)

    # removing overlaps
    for overlapping_segment in overlaps:
        print(overlapping_segment  )


    starting_time = 0
    without_overlaps_audio = AudioSegment.empty()
    for overlaping_segments in overlaps['content']:
        segment = overlaping_segments['segment']
        current_segment_start_time = segment['start'] * 1000
        current_segment_end_time = segment['end'] * 1000
        current_segment = audio[starting_time: current_segment_start_time]
        starting_time = current_segment_end_time
        
        without_overlaps_audio += current_segment
    
    without_overlaps_audio.export("temp.wav", format="wav")
    
    segments = extract_speak_segments('temp.wav')
 
    embeddings = []
        
    inference = Inference("pyannote/embedding", window="whole")
    
    timeline = Timeline()
    count_short = 0
    for segment in segments['content']:
        current_segment_time = segment['segment']
        start_time = current_segment_time['start']
        end_time = current_segment_time['end']

        if end_time - start_time < 1:
            count_short += 1
            print(str(count_short) + ' short segments')
            continue
        current_segment = Segment(start_time, end_time)
        print('Long segment: ', current_segment)
        #embedding = inference.crop("temp.wav", current_segment)
        embedding = extract_speaker_embedding(current_segment, audio)
        
        
        timeline.add(current_segment)
        embeddings.append(embedding)
    
    embeddings = np.vstack(embeddings)
    
    y = classify_speakers_based_on_embeddings(embeddings)

    speaker1= AudioSegment.empty()
    speaker2 = AudioSegment.empty()

    for i, segment in enumerate(timeline):
        current_audio_segment = audio[segment.start*1000:segment.end*1000]
        if y[i] == 0:


            speaker1 += current_audio_segment
        else:
            speaker2 += current_audio_segment

    
    speaker1.export("speaker1_old.wav", format="wav")
    speaker2.export("speaker2_old.wav", format="wav")
    
            

    return embeddings

# model_ids = {
#         "hubert": "facebook/hubert-large-ls960-ft",
#         "wav2vec2": "facebook/wav2vec2-base-960h",
# }

# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# def get_audio_embeddings(audio_array, sample_rate, model_id="facebook/wav2vec2-base-960h"):
#     input_values = processor(audio_array, sampling_rate=sample_rate, return_tensors="pt").input_values
#     output = model(**input_values)
#     hidden_states = output.last_hidden_state
#     return hidden_states


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

