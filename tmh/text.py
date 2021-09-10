from transformers import pipeline

def generate_text(model='birgermoell/swedish-gpt', prompt="AI har möjligheten att", min_length=50):
    generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')
    output = generator(prompt, do_sample=True, min_length=min_length)
    return output[0]['generated_text']
    
# output = generate_text(model='birgermoell/swedish-gpt', prompt="AI har möjligheten att", min_length=150)
# output = generate_text(model='EleutherAI/gpt-neo-2.7B', prompt="EleutherAI has", min_length=150)
# print(output)