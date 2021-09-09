"""
Tests for Text-To-Speech (TTS) functionality.

"""
from tmh.speech.tacotron import Tacotron2
import unittest
import os


class TestFlows(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.text_sentences = [{
            "text": "Room 509 is full of awesome people!",
            "filename": "509.wav"
        }]

        cls.tacotron2 = Tacotron2()

    def test_tacotron2(self):
        """
        Test tacotron 2 speech synthesis.
        """

        for text in self.text_sentences:
            self.tacotron2.synthesize(text["text"], text["filename"])
            self.assertTrue(os.path.exists(
                os.path.join(os.getcwd(), text["filename"])))
            os.remove(os.path.join(os.getcwd(), text["filename"]))
