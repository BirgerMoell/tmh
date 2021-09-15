from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchaudio
import numpy as np

def get_audio_embeddings(audio_path, model_id="facebook/wav2vec2-base-960h"):
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        model = Wav2Vec2Model.from_pretrained(model_id)
        waveform, sample_rate = torchaudio.load(audio_path)

        # wav_file = np.random.random((1, 1024))
        import pdb
        pdb.set_trace()

        input_values = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True).input_values
        hidden_states = model(input_values).last_hidden_state
        return hidden_states

audio_embeddings = get_audio_embeddings('test.wav')