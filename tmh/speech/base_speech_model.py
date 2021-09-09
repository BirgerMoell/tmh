import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseSpeechModel(nn.Module, ABC):
    """
    Base class for speech model.
    """

    def __init__(self):
        super(BaseSpeechModel, self).__init__()
        self._model = None
        self._vocoder = None

    @abstractmethod
    def load_model(self, path):
        """
        Load model from path.
        """
        raise NotImplementedError("No model loaded")

    @abstractmethod
    def load_vocoder(self, path):
        """
        Load vocoder from path.
        """
        raise NotImplementedError("No vocoder loaded")

    @abstractmethod
    def synthesize(self, text, out_path):
        """
        Main method to run the synthesis
        """
        raise NotImplementedError("No synthesize implemented")

    def push_to_cpu(self, data):
        r"""
        Pushes torch.Tensor to cpu

        Args:
            data (torch.tensor): input tensor

        Returns:
            torch.tensor: tensor ensured to be on cpu
        """
        if data.device != 'cpu':
            data = data.to('cpu')
        return data
