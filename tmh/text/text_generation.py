from transformers import pipeline

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

# output = generate_text(model='birgermoell/swedish-gpt', prompt="AI har möjligheten att", min_length=250)
# output = generate_text(model='EleutherAI/gpt-neo-2.7B', prompt="EleutherAI has", min_length=150)
# print(output)
# print(list_models())