import requests
from typing import List, Dict

class Llama3Interface:
    def __init__(self, api_base_url="http://localhost:1234/v1", api_key="lm-studio"):
        self.api_base_url = api_base_url
        self.api_key = api_key

    def generate_pairs(self, prompt: str) -> List[Dict[str, str]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }

        response = requests.post(f"{self.api_base_url}/chat/completions", headers=headers, json=data)

        if response.status_code != 200:
            return []

        result = response.json()

        # Extract the response content
        if "choices" in result and result["choices"]:
            content = result["choices"][0]["message"]["content"]
            return [{"question": prompt, "answer": content}]
        
        return []

