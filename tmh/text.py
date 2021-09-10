from transformers import pipeline

def generate_text(model='EleutherAI/gpt-neo-2.7B', prompt="EleutherAI has", min_length=50):
    generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')
    generator(prompt, do_sample=True, min_length=min_length)
    print(generator)
    return generator
    
generate_text()