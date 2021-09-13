from espnet2.bin.tts_inference import Text2Speech
    
model = Text2Speech.from_pretrained("julien-c/ljspeech_tts_train_tacotron2_raw_phn_tacotron_g2p_en_no_space_train")

speech, *_ = model("text to generate speech from")
print(speech)