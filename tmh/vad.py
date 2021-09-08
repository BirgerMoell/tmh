import webrtcvad
vad = webrtcvad.Vad()

def is_silent(data):
    return vad.is_speech(data, 16000)

# Run the VAD on 10 ms of silence. The result should be False.
sample_rate = 16000
frame_duration = 10  # ms
frame = b'\x00\x00' * int(sample_rate * frame_duration / 1000)
print ('Contains speech: %s' % (vad.is_speech(frame, sample_rate)))