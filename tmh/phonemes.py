from dp.phonemizer import Phonemizer
from dp.model import model, predictor
import torch

# this model needs a download of the model.
## Please download this model and put it in the following folder.
## https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_ipa_forward.pt
## The model assumes that the model is stored in the current folder.

def get_swedish_phonemes(text, model_path):
    checkpoint = model_path
    transformer = model.ForwardTransformer(55, 52, d_fft=1024, d_model=512, dropout=0.1, heads=4, layers=6)
    device = torch.device('cpu')
    checkpoint = torch.load(checkpoint, map_location=device)
    transformer.load_state_dict(checkpoint['model'])
    preprocessor = checkpoint['preprocessor']
    pred = predictor.Predictor(transformer, preprocessor)
    phoneme_dict = checkpoint['phoneme_dict']
    phonemize = Phonemizer(pred, phoneme_dict)
    phonemes = phonemize(text, 'se')
    result = ''
    for i, ph in enumerate(phonemes):
        if ph == ' ':
            result = result + '_ '
        if ph not in '23:_ ':
            result = result + ph + ' '
        elif ph != ' ':
            result = result[:-1] + ph + ' ' 
    return result

def get_phonemes(text, model_checkpoint='./en_us_cmudict_ipa_forward.pt', language='English'):    
    if language == 'English':
        phonemizer = Phonemizer.from_checkpoint(model_checkpoint)
        phonemes = phonemizer(text, lang='en_us')
        return phonemes
    if language == 'Swedish':
        return get_swedish_phonemes(text, model_checkpoint)





banan = get_phonemes('Välkommen till tal, musik och hörsel','best_model.pt', 'Swedish')
print(banan)