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

### Transcribe with VAD
``` python
from tmh.transcribe_with_vad import transcribe_from_audio_path_split_on_speech
file_path = "./sv.wav"
print("creating transcription")
asr_transcription_with_vad = transcribe_from_audio_path_split_on_speech(file_path)
print("transcription")
print(asr_transcription_with_vad)
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

### Speech Generation
#### Tacotron 2
Make sure you install these packages before running tacotron 2
```bash
pip install numpy scipy librosa unidecode inflect librosa
apt-get update
apt-get install -y libsndfile1
```

### Codex
Generate code and save to file.
To use
```
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



### Github
https://gits-15.sys.kth.se/bmoell/tmh