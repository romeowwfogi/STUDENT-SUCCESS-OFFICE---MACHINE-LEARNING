import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Global Variables ---
# We will train these models once when the app starts
model = LinearRegression()
scaler = StandardScaler()
pca = PCA(n_components=5)
feature_columns = []
is_trained = False
last_df = None

def train_model(file):
    df = pd.read_csv(file)
    global feature_columns
    Y = df["Total_Admitted"]
    X = df.drop(columns=["Total_Admitted", "Year"])
    feature_columns = X.columns.tolist()
    global scaler
    X_scaled = scaler.fit_transform(X)
    global pca
    X_pca = pca.fit_transform(X_scaled)
    global model
    model.fit(X_pca, Y)
    global is_trained
    is_trained = True
    global last_df
    last_df = df


# --- Flask App Initialization ---
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train_predict_next_year', methods=['POST'])
def train_predict_next_year():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        df = pd.read_csv(file)
        required = [
            'Year','Unemployment_Rate','Num_Competing_Schools','Applicants_Business_Accountancy','Applicants_Nursing',
            'Applicants_Tuition_Based','Applicants_Hospitality','Applicants_Engineering','Applicants_Computer_Studies',
            'Applicants_Education','Applicants_Within_City','Applicants_Outside_City','Admission_Rate_Last_Year','Total_Admitted'
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return jsonify({"error": f"Missing columns: {missing}"}), 400
        global feature_columns
        Y = df['Total_Admitted']
        X = df.drop(columns=['Total_Admitted','Year'])
        feature_columns = X.columns.tolist()
        global scaler
        X_scaled = scaler.fit_transform(X)
        global pca
        X_pca = pca.fit_transform(X_scaled)
        global model
        model.fit(X_pca, Y)
        global is_trained
        is_trained = True
        if 'Year' in df.columns:
            last_year_str = str(df['Year'].iloc[-1])
            try:
                start = int(str(last_year_str).split('-')[0])
                next_start = start + 1
                next_end = next_start + 1
                next_year_str = f"{next_start}-{next_end}"
            except Exception:
                next_year_str = None
        else:
            next_year_str = None
        next_features = {}
        for col in feature_columns:
            if len(df[col]) >= 2 and pd.api.types.is_numeric_dtype(df[col]):
                diff = df[col].iloc[-1] - df[col].iloc[-2]
                val = df[col].iloc[-1] + diff
            else:
                val = df[col].iloc[-1]
            if pd.api.types.is_numeric_dtype(df[col]):
                if str(col).lower().startswith('applicants') or str(col).lower().endswith('_city') or str(col).lower().endswith('schools'):
                    val = max(val, 0)
            next_features[col] = float(val) if isinstance(val, (int, float, np.number)) else val
        input_df = pd.DataFrame([next_features], columns=feature_columns)
        input_scaled = scaler.transform(input_df)
        input_pca = pca.transform(input_scaled)
        prediction = model.predict(input_pca)
        return jsonify({
            "year": next_year_str,
            "total_admitted_prediction": float(prediction[0])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    """
    The main prediction endpoint.
    Expects JSON data with keys matching the feature columns.
    """
    try:
        if not is_trained:
            return jsonify({"error": "Model not trained. Upload a CSV to /upload."}), 400
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Create a DataFrame from the input JSON
        # This ensures the features are in the correct order as defined by feature_columns
        input_df = pd.DataFrame([data], columns=feature_columns)

        # Check for missing values (if user didn't provide all features)
        if input_df.isnull().values.any():
            missing_cols = input_df.columns[input_df.isnull().any()].tolist()
            return jsonify({"error": f"Missing feature(s) in input data: {missing_cols}"}), 400

        # --- Apply the FULL prediction pipeline ---
        # 1. Scale the input data using the *fitted* scaler
        input_scaled = scaler.transform(input_df)

        # 2. Apply PCA using the *fitted* PCA object
        input_pca = pca.transform(input_scaled)

        # 3. Make a prediction using the *fitted* model
        prediction = model.predict(input_pca)

        # Return the prediction as JSON
        return jsonify({
            "total_admitted_prediction": prediction[0]
        })

    except Exception as e:
        # Generic error handler
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# --- Run the App ---
if __name__ == "__main__":
    app.run(debug=True, port=5000)