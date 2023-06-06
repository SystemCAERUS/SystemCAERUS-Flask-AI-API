from flask import Flask, jsonify, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

#data = pd.read_excel('dataset.xlsx')
data = pd.read_excel('book.xlsx')
df = data[['Department','Machine','Repair done by2']]
df = df.dropna()

machine_counts = data['Machine'].value_counts()
machines_to_remove = machine_counts[machine_counts < 20].index
data = data[~data['Machine'].isin(machines_to_remove)]

dep_counts = data['Department'].value_counts()
dep_to_remove = dep_counts[dep_counts < 20].index
data = data[~data['Department'].isin(dep_to_remove)]

emp_counts = data['Repair done by2'].value_counts()
emp_to_remove = emp_counts[emp_counts<10].index
data = data[~data['Repair done by2'].isin(emp_to_remove)]

data = data[data['Repair done by2'].notnull()]
data = data[~data['Repair done by2'].isin(['Chinthaka', 'Gihan','chinthaka','Sahan'])]

# Verify unique values in the filtered target variable
unique_values = data['Repair done by2'].unique()
print(unique_values)

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Decode the predicted labels back to original names
y_pred_decoded = label_encoder.inverse_transform(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


def predict_repairer(department, machine, clf, label_encoder, training_columns):
    # Prepare the input data for prediction
    input_data = pd.DataFrame({'Department': [department], 'Machine': [machine]})
    input_data_encoded = pd.get_dummies(input_data)
    input_data_encoded = input_data_encoded.reindex(columns=training_columns, fill_value=0)
    predicted_label = clf.predict(input_data_encoded)
    return predicted_label[0]


@app.route('/predict', methods=['POST'])
def predict():
    request_data = request.get_json()
    department = request_data['department']
    machine = request_data['machine']

    # Make predictions using the trained model
    predicted_label = predict_repairer(department, machine, clf, label_encoder, training_columns)
    predicted_repairer = label_encoder.inverse_transform([predicted_label])[0]

    # Return the predicted repairer as the response
    response = {'predicted_repairer': predicted_repairer}
    return jsonify(response)


if __name__ == '__main__':
    app.run()
