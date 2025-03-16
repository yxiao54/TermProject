from dataloader import get_data
from tqdm import tqdm
import requests



def query_ollama(question, model="mistral"):
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": question, "stream": False}

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()["response"].strip()
        return "ERROR"
    except Exception as e:
        print(f"Error querying Ollama: {e}")
        return "ERROR"
        
def evaluate(dataset, model="mistral"):
    correct = 0
    total = len(dataset)

    for question, answer,paragraphs,table in tqdm(dataset, desc="Evaluating"):
        print(question)
        print(answer)
        print(table)
   
        llm_response = query_ollama(question, model)
        if llm_response.lower() == str(answer).lower():  # Case-insensitive match
            correct += 1

    accuracy = (correct / total) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy


if __name__ == "__main__": 
    dataset=get_data(name='finqa')#finqa,tatqa
    evaluate(dataset, model="mistral")


