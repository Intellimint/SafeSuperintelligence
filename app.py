from flask import Flask, request, jsonify
import joblib
import numpy as np
from model import predict
from scraper import scrape_webpage

app = Flask(__name__)

conversation_history = []

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.json
    input_data = np.array(data['input']).reshape(1, -1)
    prediction = predict(input_data)
    return jsonify({'prediction': prediction.tolist()})

@app.route('/scrape', methods=['POST'])
def scrape_route():
    data = request.json
    url = data['url']
    content = scrape_webpage(url)
    return jsonify({'content': content})

@app.route('/conversation', methods=['POST'])
def conversation_route():
    global conversation_history
    user_input = request.json['user_input']
    conversation_history.append(f"User: {user_input}")
    
    # Process user input and generate response
    response, reflection = process_conversation(user_input, conversation_history)
    
    conversation_history.append(f"AI: {response}")
    
    return jsonify({'response': response, 'learning_reflection': reflection})

def process_conversation(user_input, conversation_history):
    # Analyze the input for context, intent, and any specific questions or requests
    if "how are you" in user_input.lower():
        response = "I'm an AI, so I don't have feelings, but I'm here to help you!"
    elif "hello" in user_input.lower():
        response = "Hello! How can I assist you today?"
    else:
        response = f"You said: {user_input}"
    
    # Reference the conversation history to maintain continuity
    if len(conversation_history) > 1:
        response += f" (Based on our previous conversation: {conversation_history[-2]})"
    
    # Generate a reflection
    reflection = "Reflecting on the interaction and learning from it."
    
    return response, reflection

feedback_history = []

@app.route('/feedback', methods=['POST'])
def feedback_route():
    global feedback_history
    feedback = request.json['feedback']
    response = request.json['response']
    feedback_history.append({'response': response, 'feedback': feedback})
    
    # Process feedback and update AI's knowledge base
    process_feedback(response, feedback)
    
    return jsonify({'status': 'Feedback received'})

def process_feedback(response, feedback):
    # Implement logic to use feedback for improving AI's responses
    # For now, we'll just print the feedback
    print(f"Received feedback: {feedback} for response: {response}")
    
    # Example: Adjust AI's behavior based on feedback
    if "great" in feedback.lower():
        print("Positive feedback received. Reinforcing current behavior.")
    elif "bad" in feedback.lower():
        print("Negative feedback received. Adjusting behavior.")


if __name__ == '__main__':
    app.run(port=5000)

