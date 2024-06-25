from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'model.pkl')

# Function to predict
def predict(input_data):
    model = joblib.load('model.pkl')
    return model.predict(input_data)

