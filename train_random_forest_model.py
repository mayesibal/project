
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load your dataset (convert your Excel to CSV as 'student_data.csv')
df = pd.read_csv("student_data.csv")
X = df.iloc[:, :3]  # First 3 columns: Admission, Past GPA, First-Year GPA
y = df.iloc[:, 3]   # Target column: Final GPA

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model to disk
joblib.dump(model, "random_forest_model.pkl")
