import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from flask import Flask, request, jsonify, render_template
from io import StringIO
import traceback
from flask_cors import CORS

# --- Global Variables ---
model = None
encoder = LabelEncoder()
is_model_trained = False
cached_metrics = {}

# Define the exact columns required for training
REQUIRED_COLUMNS = set([
    'College_Name',
    'Academic_Year',
    'Unemployment_Rate',
    'Num_Competing_Schools',
    'Admission_Rate_Last_Year',
    'Total_Admitted' 
])

# Define the features the model expects
MODEL_FEATURES = [
    'College_ID',
    'Academic_Year_Num',
    'Unemployment_Rate',
    'Num_Competing_Schools',
    'Admission_Rate_Last_Year'
]

app = Flask(__name__)
CORS(app)

def preprocess_and_train(df):
    """
    Performs data preprocessing, training, and evaluation.
    Updates the global 'model' and 'encoder' variables.
    """
    global model, encoder, is_model_trained

    # 1. Label Encoding for College_Name
    try:
        # Fit encoder on the CSV data
        df['College_ID'] = encoder.fit_transform(df['College_Name'])
    except Exception as e:
        return None, f"Error during College_Name encoding: {str(e)}"

    # 2. Feature Engineering
    try:
        df['Academic_Year_Num'] = df['Academic_Year'].astype(str).str[:4].astype(int)
    except Exception as e:
        return None, f"Error parsing 'Academic_Year' column: {str(e)}. Ensure format is 'YYYY–YYYY'."

    # 3. Prepare features (X) and target (y)
    X = df[MODEL_FEATURES]
    y = df['Total_Admitted']

    # 4. Train
    # Note: We use all data for training since we retrain on every request or use cached model
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X, y)
    is_model_trained = True

    # Calculate simple metrics on the training set just for display
    predictions = model.predict(X)
    r2 = r2_score(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))

    return {
        "rmse": float(rmse),
        "r2_score": float(r2),
        "total_records": len(df)
    }, None


@app.route('/predict', methods=['POST'])
def get_prediction():
    """
    Unified Endpoint:
    1. Checks if a 'dataset_file' is provided. If so, TRAINS the model first.
    2. PREDICTS using the input fields.
    """
    global cached_metrics

    # --- STEP 1: Check for Training Data ---
    if 'dataset_file' in request.files:
        file = request.files['dataset_file']
        if file.filename != '':
            try:
                # Train the model immediately
                csv_data = StringIO(file.read().decode('utf-8'))
                df = pd.read_csv(csv_data)
                
                metrics, error = preprocess_and_train(df)
                if error: return jsonify({"error": error}), 500
                cached_metrics = metrics
            except Exception as e:
                return jsonify({"error": f"Failed to process CSV: {str(e)}"}), 500
    
    # Check if model exists (either from previous run or just trained above)
    if not is_model_trained or model is None:
        return jsonify({"error": "Model is not trained. Please include a CSV file in your request."}), 400

    # --- STEP 2: Perform Prediction ---
    try:
        # When using FormData (multipart/form-data), we use request.form instead of request.get_json()
        data = request.form 

        required_inputs = ['College_Name', 'Academic_Year', 'Unemployment_Rate', 'Num_Competing_Schools', 'Admission_Rate_Last_Year']
        for key in required_inputs:
            if key not in data:
                return jsonify({"error": f"Missing required input parameter: '{key}'"}), 400

        college_name = data['College_Name']
        academic_year = data['Academic_Year']
        unemployment_rate = float(data['Unemployment_Rate'])
        num_competing_schools = int(data['Num_Competing_Schools'])
        admission_rate_last_year = float(data['Admission_Rate_Last_Year'])

        # Encode College Name safely
        try:
            if college_name not in encoder.classes_:
                return jsonify({"error": f"College Name '{college_name}' not found in the uploaded CSV."}), 400
            college_id = encoder.transform([college_name])[0]
        except ValueError:
            return jsonify({"error": f"College Name encoding error."}), 400

        try:
            academic_year_num = int(str(academic_year)[:4])
        except Exception:
            return jsonify({"error": "Invalid Academic Year format"}), 400

        input_features = [
            college_id,
            academic_year_num,
            unemployment_rate,
            num_competing_schools,
            admission_rate_last_year
        ]

        predicted_admissions = model.predict(np.array([input_features]))[0]
        rounded_prediction = int(np.round(predicted_admissions))

        return jsonify({
            "status": "success",
            "predicted_total_admitted": max(0, rounded_prediction),
            "college_name": college_name,
            "academic_year": academic_year,
            "training_metrics": cached_metrics
        }), 200

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)