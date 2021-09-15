from transformers import pipeline
from transformers import AutoTokenizer, AutoModelWithLMHead

def get_sentiment(text):
    classifier = pipeline('sentiment-analysis')
    sentiment = classifier('We are very happy to introduce pipeline to the transformers repository.')
    return sentiment

# sentiment = get_sentiment("Robots are the best")
# print(sentiment)

def get_emotion(text):
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")
    model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-emotion")
    input_ids = tokenizer.encode(text + '</s>', return_tensors='pt')
    output = model.generate(input_ids=input_ids,
               max_length=2)
    dec = [tokenizer.decode(ids) for ids in output]
    label = dec[0]
    return label
  
# emotion = get_emotion("i feel as if i havent blogged in ages are at least truly blogged i am doing an update cute") # Output: 'joy'
# print(emotion)
# emotion2 = get_emotion("i have a feeling i kinda lost my best friend") # Output: 'sadness'
# print(emotion2)