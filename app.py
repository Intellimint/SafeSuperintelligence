from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging
from model import predict
from scraper import scrape_webpage
from q_learning import QLearningAgent

app = Flask(__name__)

conversation_history = []
feedback_history = []
agent = QLearningAgent()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

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
    try:
        state = response
        action = feedback
        reward = 1 if "great" in feedback.lower() else -1

        # Update Q-table
        agent.update_q_value(state, action, reward, state, [action])

        # Save the Q-table after each update
        agent.save_q_table()

        # Log feedback
        logging.debug(f"Received feedback: {feedback} for response: {response}")
        if reward == 1:
            logging.debug("Positive feedback received. Reinforcing current behavior.")
        else:
            logging.debug("Negative feedback received. Adjusting behavior.")
    except Exception as e:
        logging.error(f"Error processing feedback: {e}")

if __name__ == '__main__':
    app.run(port=5000)
