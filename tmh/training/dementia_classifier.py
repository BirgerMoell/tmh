import torchaudio


data = []

control_path = "/Users/bmoell/Data/media.talkbank.org/dementia/English/Pitt/Control/cookie"
dementia_path = "/Users/bmoell/Data/media.talkbank.org/dementia/English/Pitt/Dementia/cookie"


for path in tqdm(Path("/content/data/aesdd").glob("**/*.wav")):
    name = str(path).split('/')[-1].split('.')[0]
    label = str(path).split('/')[-2]
    
    try:
        # There are some broken files
        s = torchaudio.load(path)
        data.append({
            "name": name,
            "path": path,
            "emotion": label
        })
    except Exception as e:
        # print(str(path), e)
        pass

    # break