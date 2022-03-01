from dp.phonemizer import Phonemizer
from dp.model import model, predictor
import torch
import argparse

# this model needs a download of the model.
## Please download this model and put it in the following folder.
## https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_ipa_forward.pt
## The model assumes that the model is stored in the current folder.


parser = argparse.ArgumentParser(description='Phonemize Swedish or English text')
parser.add_argument("--model_path", metavar='M', type=str, nargs='?', default='./en_us_cmudict_ipa_forward.pt',
                    help='The model path to your saved checkpoint')
parser.add_argument('--language', metavar='L', type=str, default='English',
                    help='The language you would like to phonemize')
parser.add_argument('--stress', nargs='?', metavar='S', type=str, default="n",
                    help='Indicate whether your model includes stress mark (y or n)')
parser.add_argument('--text', metavar="T", type=str,
                    help="The text you would like to phonemize formatted as a single string")
parser.add_argument('--save_to_file', nargs='?', metavar='F', type=str, 
                    help='If you want to save the phonemized text, write the filepath')
args = parser.parse_args()

def str_to_bool(string):
    if string == "y":
        return True
    if string == "n":
        return False

model_path = args.model_path
language = args.language
stress = str_to_bool(args.stress)
text = args.text
save_path = args.save_to_file

def get_swedish_phonemes(text, model_path, stress_marks=False):
    checkpoint = model_path
    if stress_marks:
        transformer = model.ForwardTransformer(55, 55, d_fft=1024, d_model=512, dropout=0.1, heads=4, layers=6)
    else:
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

def get_phonemes(text, model_checkpoint='./en_us_cmudict_ipa_forward.pt', language='English', stress_marks=False):   
    if language == 'English':
        phonemizer = Phonemizer.from_checkpoint(model_checkpoint)
        phonemes = phonemizer(text, lang='en_us')
        return phonemes
    if language == 'Swedish' and stress_marks:
        return get_swedish_phonemes(text, model_checkpoint, stress_marks=True)
    else:
        return get_swedish_phonemes(text, model_checkpoint)

if __name__=='__main__':
    if save_path:
        phones = get_phonemes(text, model_path, language, stress)
        with open(save_path, "a", encoding="utf-8") as outfile:
            outfile.write(phones + '\n')

    else:
        print(get_phonemes(text, model_path, language, stress))

