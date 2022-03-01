import os
import openai

def generate(prompt):
    response = openai.Completion.create(
    engine="davinci-codex",
    prompt=prompt,
    temperature=0,
    max_tokens=1500,
    top_p=1,
    frequency_penalty=0.5,
    presence_penalty=0.5
    )
    return response

def generate_from_prompt(text):
    response = generate(text)
    return response.choices[0].text

def write_to_file(text, file):
    info = open(file, "w")
    info.write(text)
    info.close()

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    openai.api_key = os.getenv('OPEN_AI_API_KEY')
    
    write_to_file(generate_from_prompt('''
    Make me a sandwich.
    '''), "code.txt")
