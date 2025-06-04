import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
import matplotlib.pyplot as plt
from data_preprocessing import preprocess_data  # Your preprocessing function

def evaluate_model(model_path, data_path):
    # Load raw data
    df = pd.read_excel(data_path, header=1)
    df.rename(columns={"default payment next month": "default"}, inplace=True)
    
    # Preprocess and split
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Load model
    model = joblib.load(model_path)
    
    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()
    
    # ROC Curve
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve")
    plt.show()

if __name__ == "__main__":
    model_path = "../models/model.pkl"
    data_path = "../data/raw/default_of_credit_card_clients.xls"
    evaluate_model(model_path, data_path)
