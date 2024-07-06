from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('customer_churn_model.pkl', 'rb'))  # Replace 'churn_model.pkl' with your actual filename
scaler = pickle.load(open('customer_churn_scaler.pkl', 'rb'))  # Replace 'scaler.pkl' with your actual filename

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    CreditScore = float(request.form['CreditScore'])
    Geography = float(request.form['Geography'])
    Gender = float(request.form['Gender'])
    Age = float(request.form['Age'])
    Tenure = float(request.form['Tenure'])
    Balance = float(request.form['Balance'])
    NumOfProducts = float(request.form['NumOfProducts'])
    HasCrCard = float(request.form['HasCrCard'])
    IsActiveMember = float(request.form['IsActiveMember'])
    EstimatedSalary = float(request.form['EstimatedSalary'])

    # Prepare the input data
    input_data = np.array([[CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]])
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)

    # Map the prediction to the required labels
    prediction_label = "The customer is not churn(They will continue business)" if prediction[0] == 0 else "The customer is churn(They will not continue in business)"

    # Return the result
    return render_template('result.html', prediction=prediction_label, prediction_proba=prediction_proba[0])

if __name__ == '__main__':
    app.run(debug=True)
