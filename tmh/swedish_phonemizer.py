from dp.phonemizer import Phonemizer

def try_it_out():
    checkpoint_path = '/Users/bmoell/Code/tmh/models/swedish_phonemizer.pt'
    phonemizer = Phonemizer.from_checkpoint(checkpoint_path)
    text = 'varför händer det här'
    result = phonemizer.phonemise_list([text], lang='se')
    print(result.phonemes)
    for text, pred in result.predictions.items():
        tokens, probs = pred.phoneme_tokens, pred.token_probs
        for o, p in zip(tokens, probs):
            print(f'{o} {p}')
        tokens = ''.join(tokens)
        print(f'{text} | {tokens} | {pred.confidence}')

try_it_out()