from transformers import Wav2Vec2Processor, Wav2Vec2Model
import soundfile as sf
import numpy as np

model_ids = {
        "hubert": "facebook/hubert-large-ls960-ft",
        "wav2vec2": "facebook/wav2vec2-base-960h",
}

def get_audio_embeddings(audio_path, model_id="facebook/hubert-large-ls960-ft"):
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        model = Wav2Vec2Model.from_pretrained(model_id)
        y, sample_rate = sf.read(audio_path)
        
        with torch.no_grad():
            
            input_values = processor(y, sampling_rate=sample_rate, return_tensors="pt", padding=True).input_values
            hidden_states = model(input_values).last_hidden_state
            return hidden_states

# embed =audio_embeddings = get_audio_embeddings('/Users/bmoell/Code/test_tanscribe/sv.wav')
# print(embed)
