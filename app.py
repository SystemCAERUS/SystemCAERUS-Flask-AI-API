from flask import Flask, jsonify, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


# load the exel data set
data = pd.read_excel('dataset.xlsx')

#select the data Frame
df = data[['Department','Machine','Repair done by2']]

# Dropping null values
df = df.dropna()

#Prepare X , Y data sets
X = data[['Department', 'Machine']]
Y = data['Repair done by2']

# Perform label encoding on the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(Y)

label_mapping = pd.DataFrame({'Repairer Name': Y, 'Encoded Value': y})
label_mapping = label_mapping.drop_duplicates().sort_values('Encoded Value').reset_index(drop=True)

# Perform one-hot encoding on the features
X = pd.get_dummies(X)
training_columns = list(X.columns)


# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the classification algorithm (Decision Tree classifier in this example)
clf = DecisionTreeClassifier()

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Decode the predicted labels back to original names
y_pred_decoded = label_encoder.inverse_transform(y_pred)

# Calculate accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of the ML model is :', accuracy)


def predict_repairer(department, machine, clf, label_encoder, training_columns):
    # Prepare the input data for prediction
    input_data = pd.DataFrame({'Department': [department], 'Machine': [machine]})

    # Perform one-hot encoding on the input data
    input_data_encoded = pd.get_dummies(input_data)

    # Align the input data columns with the training columns
    input_data_encoded = input_data_encoded.reindex(columns=training_columns, fill_value=0)

    # Make predictions on the input data
    predicted_label = clf.predict(input_data_encoded)

    return predicted_label[0]

# Define the predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the request data
    request_data = request.get_json()

    # Extract the department and machine from the request data
    department = request_data['department']
    machine = request_data['machine']

    # Make predictions using the trained model
    predicted_label = predict_repairer(department, machine, clf, label_encoder, training_columns)
    predicted_repairer = label_encoder.inverse_transform([predicted_label])[0]

    # Return the predicted repairer as the response
    response = {'predicted_repairer': predicted_repairer}
    return jsonify(response)

# Run the Flask app
if __name__ == '__main__':
    app.run()
