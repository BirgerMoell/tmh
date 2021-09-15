from transformers import pipeline

def get_answer(question_with_context):
    """
    Get answer for question and context.
    """
    # Create pipeline
    question_answering_pipeline = pipeline('question-answering')

    # Get answer
    answer = question_answering_pipeline(question_with_context)

    # Return answer
    return answer

answer = get_answer({'question': 'What is the meaning of life', 'context': 'The meaning of life is to be happy'})
print(answer)