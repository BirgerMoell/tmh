from dp.phonemizer import Phonemizer

# this model needs a download of the model.
## Please download this model and put it in the following folder.
## https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_ipa_forward.pt
## The model assumes that the model is stored in the current folder.

def get_phonemes(text, language="English", model='./en_us_cmudict_ipa_forward.pt'):
    if language == "Swedish":
        phonemizer = Phonemizer.from_checkpoint(model)
        phonemes = phonemizer(text, lang='se')
        return phonemes
    else:
        phonemizer = Phonemizer.from_checkpoint(model)
        phonemes = phonemizer(text, lang='en_us')
        return phonemes

phonemes = get_phonemes("Jag äter glass", language="Swedish", model="../models/checkpoints/best_model_no_optim.pt")
# phonemes = get_phonemes("This is awesome", language="English")
print(phonemes)