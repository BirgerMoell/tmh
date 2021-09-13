import requests
import json
import random

def get_random_integer(number):
    """
    This function takes in a number and returns a random integer between 0 and that number
    """
    return random.randint(0, number)

def get_meme(keyword):
    """
    This function takes in a keyword and returns a url of a meme
    """
    url = "http://api.giphy.com/v1/gifs/search?q=" + keyword + "&api_key=dc6zaTOxFJmzC&limit=10"
    response = requests.get(url)
    data = json.loads(response.text)
    
    return data['data'][get_random_integer(10)]['images']['original']['url']


meme = get_meme("turtles")
print(meme)