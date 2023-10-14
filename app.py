from flask import Flask, request, jsonify
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Load the Iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Decision Tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

@app.route('/')
def home():
    return 'Scikit-Learn Model with Flask API'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data as JSON
        data = request.get_json()

        # Ensure the input data has the expected features
        if 'features' not in data:
            return jsonify({'error': 'Missing "features" in the input data'}), 400

        # Make predictions using the trained model
        features = data['features']
        prediction = clf.predict([features])[0]

        # Map the prediction to the corresponding class label
        target_names = iris.target_names
        predicted_class = target_names[prediction]

        return jsonify({'predicted_class': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
