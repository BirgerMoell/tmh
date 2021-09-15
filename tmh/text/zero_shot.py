from transformers import pipeline

def get_zero_shot_classification(sequence_to_classify, candidate_labels, multi_label=False):
    classifier = pipeline("zero-shot-classification",
                        model="facebook/bart-large-mnli")
    result = classifier(sequence_to_classify, candidate_labels, multi_label=multi_label)
    return result

# sequence_to_classify = "one day I will see the world"
# candidate_labels = ['travel', 'cooking', 'dancing']
# classified_label = get_zero_shot_classification(sequence_to_classify, candidate_labels)
# print(classified_label)