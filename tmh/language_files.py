
language_dict = {
    "Swedish": "KBLab/wav2vec2-large-voxrex-swedish",
    "English": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
    "Russian": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "Spanish": "facebook/wav2vec2-large-xlsr-53-spanish",
    "French": "facebook/wav2vec2-large-xlsr-53-french",
    "Persian": "m3hrdadfi/wav2vec2-large-xlsr-persian",
    "Dutch": "facebook/wav2vec2-large-xlsr-53-dutch",
    "Portugese": "facebook/wav2vec2-large-xlsr-53-portuguese",
    "Chinese": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "German": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
    "Greek": "lighteternal/wav2vec2-large-xlsr-53-greek",
    "Hindi": "theainerd/Wav2Vec2-large-xlsr-hindi",
    "Italian": "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
    "Turkish": "cahya/wav2vec2-base-turkish-artificial-cv",
    "Vietnamese": "leduytan93/Fine-Tune-XLSR-Wav2Vec2-Speech2Text-Vietnamese",
    "Catalan": "ccoreilly/wav2vec2-large-100k-voxpopuli-catala",
    "Japanese": "vumichien/wav2vec2-large-xlsr-japanese-hiragana",
    "Tamil": "vumichien/wav2vec2-large-xlsr-japanese-hiragana",
    "Indonesian": "indonesian-nlp/wav2vec2-large-xlsr-indonesian",
    "Dhivevi": "shahukareem/wav2vec2-large-xlsr-53-dhivehi",
    "Polish": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "Thai": "sakares/wav2vec2-large-xlsr-thai-demo",
    "Hebrew": "imvladikon/wav2vec2-large-xlsr-53-hebrew",
    "Mongolian": "sammy786/wav2vec2-large-xlsr-mongolian",
    "Czech": "arampacha/wav2vec2-large-xlsr-czech",
    "Icelandic": "m3hrdadfi/wav2vec2-large-xlsr-icelandic",
    "Irish": "jimregan/wav2vec2-large-xlsr-irish-basic",
    "Kinyarwanda": "lucio/wav2vec2-large-xlsr-kinyarwanda",
    "Lithuanian": "DeividasM/wav2vec2-large-xlsr-53-lithuanian",
    "Hungarian": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "Finnish": "aapot/wav2vec2-large-xlsr-53-finnish"
    }

def get_model(language):
    return language_dict[language]

language_list = "Arabic, Basque, Breton, Catalan, Chinese_China, Chinese_Hongkong, Chinese_Taiwan, Chuvash, Czech, Dhivehi, Dutch, English, Esperanto, Estonian, French, Frisian, Georgian, German, Greek, Hakha_Chin, Indonesian, Interlingua, Italian, Japanese, Kabyle, Kinyarwanda, Kyrgyz, Latvian, Maltese, Mangolian, Persian, Polish, Portuguese, Romanian, Romansh_Sursilvan, Russian, Sakha, Slovenian, Spanish, Swedish, Tamil, Tatar, Turkish, Ukranian, Welsh"
