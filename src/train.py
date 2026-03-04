import pandas as pd
import numpy as np
import logging
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(filepath: str) -> pd.DataFrame:
    """Load the Telco churn dataset from a CSV file."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Preprocess the dataset by handling missing values and encoding the target."""
    logger.info("Preprocessing data...")
    # Handle missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Convert object columns to category
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'customerID' and col != 'Churn':
            df[col] = df[col].astype('category')

    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Drop customerID as it's not a feature
    df = df.drop('customerID', axis=1)

    X = df.drop('Churn', axis=1)
    y = df['Churn']
    return X, y

def build_pipeline() -> Pipeline:
    """Build a Scikit-Learn pipeline for preprocessing and modeling."""
    logger.info("Building Scikit-Learn pipeline...")
    
    # Define feature groups
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Let's split categorical features based on number of unique values for better encoding
    # Binary/Ordinal like 'gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling'
    ordinal_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    
    # Multi-class categorical features
    onehot_features = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                       'Contract', 'PaymentMethod']

    # Preprocessing for numeric data
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Preprocessing for ordinal data
    ordinal_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    # Preprocessing for onehot data
    onehot_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('ord', ordinal_transformer, ordinal_features),
            ('cat', onehot_transformer, onehot_features)
        ])

    # Create the complete pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'))
    ])
    
    return pipeline

def train_and_evaluate(X: pd.DataFrame, y: pd.Series, output_dir: str = 'models'):
    """Train the model, evaluate metrics, and save artifacts."""
    logger.info("Splitting data for training and testing...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = build_pipeline()

    logger.info("Training the model pipeline...")
    pipeline.fit(X_train, y_train)

    logger.info("Evaluating model performance...")
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': float(pipeline.score(X_test, y_test)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    logger.info(f"Model trained successfully!")
    logger.info(f"Accuracy: {metrics['accuracy']:.3f}")
    logger.info(f"ROC-AUC: {metrics['roc_auc']:.3f}")

    # Save artifacts
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    pipeline_path = f"{output_dir}/churn_pipeline.pkl"
    metrics_path = f"{output_dir}/metrics.json"
    
    logger.info(f"Saving pipeline to {pipeline_path}")
    joblib.dump(pipeline, pipeline_path)
    
    logger.info(f"Saving metrics to {metrics_path}")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return pipeline, metrics

if __name__ == "__main__":
    try:
        data_path = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
        df = load_data(data_path)
        X, y = preprocess_data(df)
        pipeline, metrics = train_and_evaluate(X, y)
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
