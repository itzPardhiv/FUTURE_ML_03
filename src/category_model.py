import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_category_classifier(df, text_column="Resume_str", category_column="Category"):
    """
    Train a resume category classifier using LinearSVC.
    
    Args:
        df (pd.DataFrame): DataFrame containing resume texts and categories
        text_column (str): Name of column containing resume text
        category_column (str): Name of column containing category labels
        
    Returns:
        tuple: (trained_model, accuracy_score)
        
    Raises:
        ValueError: If required columns are missing or insufficient data
    """
    
    # Validate inputs
    if df is None or df.empty:
        raise ValueError("DataFrame is empty")
    
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in DataFrame")
    
    if category_column not in df.columns:
        raise ValueError(f"Category column '{category_column}' not found in DataFrame")
    
    # Filter out NaN values
    df_clean = df[[text_column, category_column]].dropna()
    
    if df_clean.empty:
        raise ValueError("No valid data after removing NaN values")
    
    if len(df_clean) < 10:
        raise ValueError(f"Insufficient data: {len(df_clean)} samples. Minimum 10 required.")
    
    X = df_clean[text_column].astype(str)
    y = df_clean[category_column].astype(str)
    
    # Verify we have at least 2 categories
    unique_categories = y.nunique()
    if unique_categories < 2:
        raise ValueError(f"Only {unique_categories} category found. Need at least 2 for classification.")
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Build and train classifier pipeline
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=1000)),
        ("clf", LinearSVC(random_state=42, max_iter=2000))
    ])
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return model, accuracy


def predict_resume_category(model, resume_text):
    """
    Predict the category of a single resume.
    
    Args:
        model: Trained classifier model
        resume_text (str): Resume text to classify
        
    Returns:
        str: Predicted category
    """
    if model is None:
        return "Unknown"
    
    try:
        prediction = model.predict([str(resume_text)])
        return prediction[0]
    except Exception as e:
        print(f"Error predicting category: {e}")
        return "Unknown"