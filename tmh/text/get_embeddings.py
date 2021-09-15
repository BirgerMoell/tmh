from transformers import BertTokenizer, BertModel
import torch

def get_bert_embedding_from_text(text, model="KB/bert-base-swedish-cased"):
    tokenizer = BertTokenizer.from_pretrained(model)
    model = BertModel.from_pretrained(model)
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states

# embedding = get_bert_embedding_from_text("Hej, jag gillar glass", model="KB/bert-base-swedish-cased")
# print(embedding)