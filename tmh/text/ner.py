from transformers import pipeline

models = {
    'kb-ner': 'KB/bert-base-swedish-cased-ner',
    "swedish-wikiann": "birgermoell/ner-swedish-wikiann"
}

def named_entity_recognition(text, model='KB/bert-base-swedish-cased-ner', tokenizer='KB/bert-base-swedish-cased-ner'):
    """
    Named entity recognition
    """
    # Load the model
    ner = pipeline('ner', model=model, tokenizer=tokenizer)
    # Get the result
    result = ner(text)
    # Return the resultß
    return result

# ner = named_entity_recognition('KTH är ett universitet i Stockholm')
# print(ner)