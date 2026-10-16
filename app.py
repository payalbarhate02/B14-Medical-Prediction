import pickle
from flask import Flask, render_template_string, request
import numpy as np

app = Flask(__name__)

# Load the trained Linear Regression model
model = pickle.load(open('medical.pkl', 'rb'))

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Cost Prediction</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-blue-50 to-indigo-100 min-h-screen flex items-center justify-center p-4">
    <div class="max-w-xl w-full bg-white rounded-2xl shadow-xl overflow-hidden p-8">
        <div class="text-center mb-8">
            <h1 class="text-3xl font-extrabold text-indigo-900">Medical Cost Predictor</h1>
            <p class="text-gray-500 mt-2">Enter patient details to estimate insurance medical charges</p>
        </div>

        <form method="POST" class="space-y-4">
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                    <label class="block text-sm font-medium text-gray-700">Age</label>
                    <input type="number" name="age" required class="mt-1 block w-full rounded-lg border-gray-300 border p-2.5 focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g. 30">
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700">Sex</label>
                    <select name="sex" class="mt-1 block w-full rounded-lg border-gray-300 border p-2.5 focus:ring-indigo-500 focus:border-indigo-500">
                        <option value="1">Male</option>
                        <option value="0">Female</option>
                    </select>
                </div>
            </div>

            <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                    <label class="block text-sm font-medium text-gray-700">BMI</label>
                    <input type="number" step="0.1" name="bmi" required class="mt-1 block w-full rounded-lg border-gray-300 border p-2.5 focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g. 25.4">
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700">Children</label>
                    <input type="number" name="children" required class="mt-1 block w-full rounded-lg border-gray-300 border p-2.5 focus:ring-indigo-500 focus:border-indigo-500" placeholder="e.g. 1">
                </div>
            </div>

            <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                    <label class="block text-sm font-medium text-gray-700">Smoker</label>
                    <select name="smoker" class="mt-1 block w-full rounded-lg border-gray-300 border p-2.5 focus:ring-indigo-500 focus:border-indigo-500">
                        <option value="1">Yes</option>
                        <option value="0">No</option>
                    </select>
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700">Region</label>
                    <select name="region" class="mt-1 block w-full rounded-lg border-gray-300 border p-2.5 focus:ring-indigo-500 focus:border-indigo-500">
                        <option value="0">Northeast</option>
                        <option value="1">Northwest</option>
                        <option value="2">Southeast</option>
                        <option value="3">Southwest</option>
                    </select>
                </div>
            </div>

            <button type="submit" class="w-full mt-6 bg-indigo-600 text-white font-semibold py-3 px-4 rounded-lg hover:bg-indigo-700 transition duration-200 shadow-md">
                Predict Charges
            </button>
        </form>

        {% if prediction is not none %}
        <div class="mt-8 p-4 bg-indigo-50 rounded-xl border border-indigo-200 text-center">
            <h2 class="text-sm font-semibold text-indigo-600 uppercase tracking-wider">Estimated Charges</h2>
            <p class="text-3xl font-bold text-indigo-900 mt-1">${{ prediction }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            age = float(request.form['age'])
            sex = float(request.form['sex'])
            bmi = float(request.form['bmi'])
            children = float(request.form['children'])
            smoker = float(request.form['smoker'])
            region = float(request.form['region'])

            # Matches the exact order of features: age, sex, bmi, children, smoker, region[cite: 2]
            features = np.array([[age, sex, bmi, children, smoker, region]])
            pred = model.predict(features)[0]
            prediction = round(pred, 2)
        except Exception as e:
            print(f"Error during prediction: {e}")
            
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
