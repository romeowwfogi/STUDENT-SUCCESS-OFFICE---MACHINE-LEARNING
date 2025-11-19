import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np # Added for np.sqrt

# Load the dataset
df = pd.read_csv("dataset.csv")

encoder = LabelEncoder()
df["College_ID"] = encoder.fit_transform(df["College_Name"])

features = [
    'College_ID',
    'Academic_Year',
    'Unemployment_Rate',
    'Num_Competing_Schools',
    'Admission_Rate_Last_Year'
]

# Convert Academic_Year to numeric (start year only)
df['Academic_Year_Num'] = df['Academic_Year'].str[:4].astype(int)
features[1] = 'Academic_Year_Num'

X = df[features]
y = df['Total_Admitted']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("RMSE:", rmse)
print("R² Score:", r2)

college_name = "BACHELOR OF SCIENCE IN INFORMATION TECHNOLOGY (BSIT)"
college_id = encoder.transform([college_name])[0]

new_data = [[
    college_id,
    2025,           # Academic Year as numeric
    5.2,            # Unemployment rate
    12,             # Number of competing schools
    0.85            # Admission rate last year
]]

predicted_admission = model.predict(new_data)[0]
print(f"Predicted admissions for {college_name} in 2025–2026:", int(predicted_admission))