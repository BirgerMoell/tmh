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

def generate_text(model='birgermoell/swedish-gpt', prompt="AI har möjligheten att", max_length=200, temperature=0.9):
    generator = pipeline('text-generation', model=model)
    output = generator(prompt, do_sample=True, temperature=temperature, max_length=max_length)
    return output[0]['generated_text']

def list_models():
    return models

def translate_and_generate(swedish_short_text, max_length=250, temperature=0.9):

    english_short_text = translate_between_languages(swedish_short_text, "Helsinki-NLP/opus-mt-sv-en")
    # print("the long english text is", english_long_text)

    english_generation = generate_text(model='EleutherAI/gpt-neo-2.7B', prompt=english_short_text, max_length=max_length, temperature=temperature)
    # print("the english summary is", english_summary)

    swedish_generation = translate_between_languages(english_generation, "Helsinki-NLP/opus-mt-en-sv")
    # print("the swedish summary is", swedish_summary)
    return swedish_generation

# generate text with translation

# output = generate_text(model='birgermoell/swedish-gpt', prompt="AI har möjligheten att", min_length=250)
# output = generate_text(model='EleutherAI/gpt-neo-2.7B', prompt="EleutherAI has", min_length=150)
# print(output)
# print(list_models())

# output = translate_and_generate('Artificiell intelligens har möjligheten att', 500)
# print(output)

# text = generate_text(prompt="Kärlek är", max_length=250, temperature=0.9)
# print("Swedish", text)
# text2 = translate_and_generate('Kärlek är', max_length=250, temperature=0.9)
# print("Translated", text2)

# text = generate_text(prompt="Kärlek är", max_length=250, temperature=0.9)
# print("Swedish", text)