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
from tmh.overalp import overlap_detection

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
phonemes = get_phonemes("I'm eating a cake", model_checkpoint='./en_us_cmudict_ipa_forward.pt')
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

output = generate_text(model='birgermoell/swedish-gpt', prompt="AI har möjligheten att", min_length=150)
print(output)
```

#### Example GPT-j

```python
from from tmh.text.text_generation import generate_text import generate_text

output = generate_text(model='EleutherAI/gpt-neo-2.7B', prompt="EleutherAI has", min_length=150)
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