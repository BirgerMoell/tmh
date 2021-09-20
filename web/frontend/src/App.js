import React, { useState } from 'react';
import logo from './logo.jpeg';
import './App.css';

// make it possible to call the api

// different things

// audio


// text

const base_url = "http://localhost:4000/"


function App() {

  const [response, setResponse] = useState("")
  const [text, setText] = useState("")

  async function generateFromApi(text, api) {
    let data = JSON.stringify({
        "text": text,
        "model": "birgermoell/swedish-gpt",
        "max_length": 250,
        "temperature": 0.9
    })

    console.log("the request is", data)
  
    let url = "http://localhost:4000/generate"
  
    let response = await fetch(url,
      {
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json"
        },
        body: data,
        method: "POST"
      })

    console.log("the response is", response)
  
    let responseJson = await response.json()
  
    console.log("the response json", responseJson)
    setResponse(responseJson.text)
    return responseJson
  }





  
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          TMH package
        </p>

        <textarea onChange={(e) => setText(e.target.value)}></textarea>
      <button onClick={()=> generateFromApi(text)}>Generate text</button>

      {response && <p className="response-text">{response}</p>}
      
      </header>
    </div>
  );
}

export default App;
