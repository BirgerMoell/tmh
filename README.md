# TMH Speech
TMH Speech is a library that gives access to open source models for transcription.

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

