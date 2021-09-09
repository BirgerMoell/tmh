# TMH Speech
TMH Speech is a library that gives access to open source models for transcription.

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
## https://huggingface.co/speechbrain/spkrec-xvect-voxceleb

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

## Build instructions
Change the version number

```
python3 -m build 
twine upload --skip-existing dist/*
```

### Github
https://gits-15.sys.kth.se/bmoell/tmh