from transformers import pipeline
from tmh.text.translate import translate_between_languages

models = {
    'gpt-j': 'EleutherAI/gpt-neo-2.7B',
    "swedish-gpt": "birgermoell/swedish-gpt",
    "swe-gpt-wiki": "flax-community/swe-gpt-wiki",
    "gpt2": "huggingface/gpt2",
    "gpt2-medium": "huggingface/gpt2-medium",
    "gpt2-large": "huggingface/gpt2-large",
    "gpt2-xl": "huggingface/gpt2-xl",
    "gpt2-xlm": "huggingface/gpt2-xlm",
    "gpt2-xlm-mlm": "huggingface/gpt2-xlm-mlm",
}

def generate_text(model='birgermoell/swedish-gpt', prompt="AI har möjligheten att", min_length=50):
    generator = pipeline('text-generation', model=model)
    output = generator(prompt, do_sample=True, min_length=min_length)
    return output[0]['generated_text']

def list_models():
    return models


def translate_and_generate(swedish_short_text, min_length=50):

    english_short_text = translate_between_languages(swedish_short_text, "Helsinki-NLP/opus-mt-sv-en")
    # print("the long english text is", english_long_text)

    english_generation = generate_text(model='EleutherAI/gpt-neo-2.7B', prompt=english_short_text, min_length=min_length)
    # print("the english summary is", english_summary)

    swedish_generation = translate_between_languages(english_generation, "Helsinki-NLP/opus-mt-en-sv")
    # print("the swedish summary is", swedish_summary)
    return swedish_generation

# generate text with translation

# output = generate_text(model='birgermoell/swedish-gpt', prompt="AI har möjligheten att", min_length=250)
# output = generate_text(model='EleutherAI/gpt-neo-2.7B', prompt="EleutherAI has", min_length=150)
# print(output)
# print(list_models())

# output = translate_and_generate('''En junimorgon då det är för tidigt
# att vakna men för sent att somna om.

# Jag måste ut i grönskan som är fullsatt
# av minnen, och de följer mig med blicken.

# ''', 150)
# print(output)