import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import numpy as np

def load_data(path):
    df = pd.read_excel(path, header=1)
    return df

def tune_hyperparameters(data_path, save_path):
    df = load_data(data_path)

    X = df.drop(['ID', 'default payment next month'], axis=1)
    y = df['default payment next month']

    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=42))
    ])

    param_dist = {
        'clf__n_estimators': [50, 100, 200, 300],
        'clf__max_depth': [None, 10, 20, 30, 40, 50],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4],
        'clf__bootstrap': [True, False]
    }

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        verbose=2,
        n_jobs=-1,
        scoring='f1',
        random_state=42
    )

    random_search.fit(X_train, y_train)

    print("Best Parameters:", random_search.best_params_)

    y_pred = random_search.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(random_search.best_estimator_, save_path)
    print(f"Best pipeline saved to {save_path}")

if __name__ == "__main__":
    data_path = "../data/raw/default_of_credit_card_clients.xls"
    save_path = "../models/best_model.pkl"
    tune_hyperparameters(data_path, save_path)
