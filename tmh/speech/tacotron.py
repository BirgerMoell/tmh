import torch
import torch.nn as nn
import torchaudio
# from BaseSpeechModel import BaseSpeechModel
from .base_speech_model import BaseSpeechModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Tacotron2(BaseSpeechModel):

    def __init__(self):
        super(Tacotron2, self).__init__()
        self.model = self.load_model()
        self.vocoder = self.load_vocoder()
        self.text_normalizer = self.get_text_utils()
        self.sample_rate = 22050

    def load_model(self):
        r"""
        Loads tacotron 2
        """
        tacotron2 = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
        tacotron2 = tacotron2.to(device)
        tacotron2.eval()
        return tacotron2

    def load_vocoder(self):
        r"""
        Loads waveglow
        """
        waveglow = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
        waveglow = waveglow.remove_weightnorm(waveglow)
        waveglow = waveglow.to(device)
        waveglow.eval()
        return waveglow

    def get_text_utils(self):
        r"""
        Download text preprocessing utils
        """
        return torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')

    def write_to_file(self, filename, data):
        r"""
        Write numpy array of audio content to file

        Args:
            filename (str): final output filename
            data (torch.tensor): audio data
        """
        data = self.push_to_cpu(data)

        torchaudio.save(filename, data, self.sample_rate)

    def synthesize(self, text, filename):
        r"""
        Main function to use for text synthesise

        Args:
            text (str): text to convert to audio
            filename (str): final output filename

        Usage:
        ```
        >>> from tmh.speech.tacotron import Tacotron2
        >>> tacotron = Tacotron2()
        >>> text = "Hello"
        >>> filename = "test.wav"
        >>> tacotron.synthesize(text, filename)
        ```
        """

        sequences, lengths = self.text_normalizer.prepare_input_sequence([
                                                                         text])

        with torch.no_grad():
            mel, _, _ = self.model.infer(sequences, lengths)
            audio = self.vocoder.infer(mel)

        self.write_to_file(filename, audio)
