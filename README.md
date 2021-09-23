# TMH Speech
TMH Speech is a library that gives access to open source models for transcription.

## Read the docs
https://tmh-docs.readthedocs.io/en/latest/docs.html#getting-started

### Getting started
To start the project you first need to install tmh and pyannote, since we are using newer packages.

```
pip install tmh
pip install https://github.com/pyannote/pyannote-audio/archive/develop.zip
```

## Example usage

### Transcription
``` python
from tmh.transcribe import transcribe_from_audio_path
file_path = "./sv.wav"
transcription = "Nu prövar vi att spela in ljud på svenska sex laxar i en laxask de finns en stor banan"
print("creating transcription")
asr_transcription = transcribe_from_audio_path(file_path)
print("output")
print(asr_transcription)
print("the transcription is", transcription)
```

### Transcribe with VAD
``` python
from tmh.transcribe_with_vad import transcribe_from_audio_path_split_on_speech
file_path = "./sv.wav"
print("creating transcription")
asr_transcription_with_vad = transcribe_from_audio_path_split_on_speech(file_path)
print("transcription")
print(asr_transcription_with_vad)
```

### Overlap detection
```python
from tmh.overlap import overlap_detection

file_path = "./sv.wav"
overlap = overlap_detection(audio_path)
print(overlap)
```

### Language classification
``` python
from tmh.transcribe import classify_language
file_path = "./sv.wav"
transcription = "Nu prövar vi att spela in ljud på svenska sex laxar i en laxask de finns en stor banan"
print("classifying language")
language = classify_language(file_path)
print("the language is", language)
```

### Classify emotion
``` python
from tmh.transcribe import classify_emotion
file_path = "./sv.wav"
print("classifying emotion")
language = classify_emotion(file_path)
print("the emotion is", language)
```
## Speaker embeddings
The speaker embeddings are made using the following library
https://huggingface.co/speechbrain/spkrec-xvect-voxceleb

### Extract speaker embedding
``` python
from tmh.transcribe import extract_speaker_embedding
file_path = "./sv.wav"
print("extracting speaker embedding")
embeddings = extract_speaker_embedding(file_path)
print("the speaker embedding is", embeddings)
```

### Extract audio embedding
Audio embeddings using wav2vec2
https://arxiv.org/abs/2006.11477
or hubert (default) https://arxiv.org/abs/2106.07447
can be used.

Note that embeddings that are not trained for ASR (using CTC) usually have better performance on classification tasks.

https://arxiv.org/pdf/2104.03502v1.pdf

From the paper, "when the model is finetuned for an ASR task, information that is not relevant for that task but might be relevant for speech emotion recognition is lost from the embeddings. For example, information about the pitch might not be important for speech recognition, while it is essential for speech emotion recognition."


``` python
from tmh.audio_embeddings import extract_audio_embeddings
audio_embeddings = get_audio_embeddings('/Users/bmoell/Code/test_tanscribe/sv.wav')
print(audio_embeddings)
```

### Voice activity detection
``` python
from tmh.vad import extract_silences
file_path = "./sv.wav"
print("extracting silences")
embeddings = extract_silences(file_path)
print("the silences are", embeddings)
```

### Phonemes
Please download this model and put it in your current folder to be able to run the model
https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_ipa_forward.pt
The model assumes that the model is stored at ./en_us_cmudict_ipa_forward.pt (you can change the model checkpoint param to save to another location)
```python
from tmh.phonemes import get_phonemes
phonemes = get_phonemes("I'm eating a cake", model_checkpoint='./en_us_cmudict_ipa_forward.pt', language="English")
print(phonemes)
```

#### Swedish phonemes
To use the swedish phonemes you need a swedish model stored at the model checkpoint path.

```python
from tmh.phonemes import get_phonemes
phonemes = get_phonemes('Välkommen till tal, musik och hörsel', model_checkpoint='swedish_model.pt', language="Swedish")
print(phonemes)
```

### Speech Generation
#### Tacotron 2
Make sure you install these packages before running tacotron 2
```bash
pip install numpy scipy librosa unidecode inflect librosa
apt-get update
apt-get install -y libsndfile1
```

## Text

### Text generation
You can use the text generation api to generate text based on any pretrained model from huggingface.

#### Example Swedish

```python
from tmh.text.text_generation import generate_text

output = generate_text(model='birgermoell/swedish-gpt', prompt="AI har möjligheten att", max_length=250, temperature=0.9)
print(output)
```

#### Example GPT-j

```python
from from tmh.text.text_generation import generate_text

output = generate_text(model='EleutherAI/gpt-neo-2.7B', prompt="EleutherAI has", max_length=250, temperature=0.9)
print(output)
```

### Exampel translate and generate
```python
from from tmh.text.text_generation import translate_and_generate

output = translate_and_generate("AI har möjligheten att skapa ett nytt samhälle där människor", max_length=250, temperature=0.9)
print(output) 
```


### Text Embeddings

```python
from tmh.text.get_embeddings import get_bert_embedding_from_text

embedding = get_bert_embedding_from_text("Hej, jag gillar glass", model="KB/bert-base-swedish-cased")
print(embedding)
```

### Named Entity Recognition

```python
from tmh.text.ner import named_entity_recognition

ner = named_entity_recognition('KTH är ett universitet i Stockholm')
print(ner)
```
### Question Answering

```python
from tmh.text.question_answering import get_answer

answer = get_answer({'question': 'What is the meaning of life', 'context': 'The meaning of life is to be happy'})
print(answer)
```

### Translation

```python
from tmh.text.translate import translate_text

sv_text = "Albert Einstein var son till Hermann och Pauline Einstein, vilka var icke-religiösa judar och tillhörde medelklassen. Fadern var försäljare och drev senare en elektroteknisk fabrik. Familjen bosatte sig 1880 i München där Einstein gick i en katolsk skola. Mängder av myter cirkulerar om Albert Einsteins person. En av dessa är att han som barn skulle ha haft svårigheter med matematik, vilket dock motsägs av hans utmärkta betyg i ämnet.[15] Han nämnde ofta senare att han inte trivdes i skolan på grund av dess pedagogik. Att Albert Einstein skulle vara släkt med musikvetaren Alfred Einstein är ett, ofta framfört, obevisat påstående. Alfred Einsteins dotter Eva har framhållit att något sådant släktskap inte existerar."

translation = translate_text(sv_text)
print(translation)
```

### Zero Shot Classification

```python
from tmh.text.zero_shot import get_zero_shot_classification

sequence_to_classify = "one day I will see the world"
candidate_labels = ['travel', 'cooking', 'dancing']
classified_label = get_zero_shot_classification(sequence_to_classify, candidate_labels)
print(classified_label)
```
### Summary

```python
from tmh.text.summarization import get_summary

ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
 A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
 Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
 In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
 Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
 2010 marriage license application, according to court documents.
 Prosecutors said the marriages were part of an immigration scam.
 On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
 After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
 Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
 All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
 Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
 Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
 The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
 Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
 Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
 If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
 """

sum = get_summary(ARTICLE)
print(sum)
```

### Translate and summarize

```python
from tmh.text.summarization import translate_and_summarize
sv_text = "Albert Einstein var son till Hermann och Pauline Einstein, vilka var icke-religiösa judar och tillhörde medelklassen. Fadern var försäljare och drev senare en elektroteknisk fabrik. Familjen bosatte sig 1880 i München där Einstein gick i en katolsk skola. Mängder av myter cirkulerar om Albert Einsteins person. En av dessa är att han som barn skulle ha haft svårigheter med matematik, vilket dock motsägs av hans utmärkta betyg i ämnet.[15] Han nämnde ofta senare att han inte trivdes i skolan på grund av dess pedagogik. Att Albert Einstein skulle vara släkt med musikvetaren Alfred Einstein är ett, ofta framfört, obevisat påstående. Alfred Einsteins dotter Eva har framhållit att något sådant släktskap inte existerar."

swedish_summary = translate_and_summarize(sv_text)
print(swedish_summary)
```

### Sentiment Analysis

```python
from tmh.text.sentiment_analysis import get_sentiment

sentiment = get_sentiment("Robots are the best")
print(sentiment)
```
### Emotion classification

```python
from tmh.text.sentiment_analysis import get_emotion

emotion = get_emotion("i feel as if i havent blogged in ages are at least truly blogged i am doing an update cute")
print(emotion)
```

### Codex
Generate code and save to file.
To use
```python
from tmh.code import generate_from_prompt, write_to_file
response = generate_from_prompt('''
A pytorch neural network model for MNIST
'''
)
write_to_file(response, "generated.py")
```

## Build instructions
Change the version number

```
python3 -m build 
twine upload --skip-existing dist/*
```

### Read the docs
https://tmh-docs.readthedocs.io/en/latest/docs.html#getting-started

### Github
https://github.com/BirgerMoell/tmh
