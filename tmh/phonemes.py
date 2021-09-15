from dp.phonemizer import Phonemizer

# this model needs a download of the model.
## Please download this model and put it in the following folder.
## https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_ipa_forward.pt
## The model assumes that the model is stored in the current folder.

def get_phonemes(text, model_checkpoint='./en_us_cmudict_ipa_forward.pt'):    
    phonemizer = Phonemizer.from_checkpoint(model_checkpoint)
    phonemes = phonemizer(text, lang='en_us')
    # print(phonemes)
    return phonemes

# get_phonemes("I'm eating a cake")