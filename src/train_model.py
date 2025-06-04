import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import os

def load_data(path):
    df = pd.read_excel(path, header=1)
    return df

def preprocess_and_train(data_path, model_path):
    df = load_data(data_path)

    # Drop ID and separate target
    X = df.drop(['ID', 'default payment next month'], axis=1)
    y = df['default payment next month']

    # Check for missing values
    if X.isnull().sum().sum() > 0:
        print("Missing values found! Filling with median...")
        X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Build pipeline: scaler + classifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Pipeline saved to {model_path}")

if __name__ == "__main__":
    data_path = "../data/raw/default_of_credit_card_clients.xls"
    model_path = "../models/model.pkl"
    preprocess_and_train(data_path, model_path)
    
