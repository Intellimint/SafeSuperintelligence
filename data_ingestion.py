import json
import os
from scraper import extract_text_from_url

def update_dataset(new_data, dataset_path='dynamic_conversations.json'):
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r') as file:
            dataset = json.load(file)
    else:
        dataset = []

    dataset.extend(new_data)

    with open(dataset_path, 'w') as file:
        json.dump(dataset, file)

def fetch_and_update_data(urls):
    new_data = []
    for url in urls:
        text = extract_text_from_url(url)
        if not text.startswith("Error"):
            new_data.append(text)
    update_dataset(new_data)

# Example usage:
urls = ['http://example.com', 'http://another-example.com']
fetch_and_update_data(urls)
