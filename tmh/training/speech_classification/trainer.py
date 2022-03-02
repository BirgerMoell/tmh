import torchaudio
import tqdm
from path import Path
import glob
from tmh.transcribe import transcribe_from_audio_path
import csv
import numpy as np

data = []

control_path = "/mnt/cloud/data/dementia/media.talkbank.org/dementia/English/Pitt/Control/cookie"
dementia_path = "/mnt/cloud/data/dementia/media.talkbank.org/dementia/English/Pitt/Dementia/cookie"

## Creata a csv file with path and label and transcription

def compute_cepstrum(signal, sample_freq):
    """Computes cepstrum."""
    frame_size = signal.size
    windowed_signal = np.hamming(frame_size) * signal
    dt = 1/sample_freq
    freq_vector = np.fft.rfftfreq(frame_size, d=dt)
    X = np.fft.rfft(windowed_signal)
    log_X = np.log(np.abs(X))
    cepstrum = np.fft.rfft(log_X)
    df = freq_vector[1] - freq_vector[0]
    quefrency_vector = np.fft.rfftfreq(log_X.size, df)
    return quefrency_vector, cepstrum

def cepstrum_f0_detection(signal, sample_freq, fmin=82, fmax=640):
    """Returns f0 based on cepstral processing."""
    quefrency_vector, cepstrum = compute_cepstrum(signal, sample_freq)
    # extract peak in cepstrum in valid region
    valid = (quefrency_vector > 1/fmax) & (quefrency_vector <= 1/fmin)
    max_quefrency_index = np.argmax(np.abs(cepstrum)[valid])
    f0 = 1/quefrency_vector[valid][max_quefrency_index]
    return f0


def load_files(file_path, label, csv_file_name):

    f = open(csv_file_name, 'w')
    writer = csv.writer(f)
    header = ['file_name', 'label', 'transcription', 'quefrency_vector', 'cepstrum', 'cepstrum_f0']
    writer.writerow(header)

    for file_name in glob.glob(file_path + "/*.mp3"):

        audio_file, sample_rate = torchaudio.load(file_name)
        np_arr = audio_file[0].cpu().detach().numpy()
        quefrency_vector, cepstrum = compute_cepstrum(np_arr, sample_rate)
        cepstrum_f0 = cepstrum_f0_detection(np_arr, sample_rate)

        print(file_name)

        transcription = transcribe_from_audio_path(file_name, model="jonatasgrosman/wav2vec2-large-xlsr-53-english")
        print(transcription)

        data = [file_name, label, transcription, quefrency_vector, cepstrum, cepstrum_f0]

        # write the data
        writer.writerow(data)

load_files(control_path, 1, "dementia.csv")