"""
FakeGuard AI - Unified Fake Account Detection Model
Detects fake accounts across GitHub, LinkedIn, and Instagram using XGBoost
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import pickle
import json
import re
import ast


def load_linkedin_data(filepath):
    """Load and preprocess LinkedIn dataset"""
    df = pd.read_csv(filepath)
    
    # The 'test' column contains values: 0, 1, 10, 11
    # We need to map these to binary: Real (1) vs Fake (0)
    # Based on typical patterns: 0 and 1 are likely real accounts, 10 and 11 are likely fake
    # Or we can interpret: values >= 10 as fake, < 10 as real
    if 'test' in df.columns:
        # Map to binary: 0 or 1 -> Real (1), 10 or 11 -> Fake (0)
        df['label'] = df['test'].apply(lambda x: 0 if x >= 10 else 1)
        df = df.drop(columns=['test'])
    
    # Extract numeric features from the dataset
    features = {}
    
    # Numeric columns that are already clean
    numeric_cols = ['Connections', 'Followers', 'Number of Experiences', 'Number of Educations',
                    'Number of Licenses', 'Number of Volunteering', 'Number of Skills',
                    'Number of Recommendations', 'Number of Projects', 'Number of Publications',
                    'Number of Courses', 'Number of Honors', 'Number of Scores',
                    'Number of Languages', 'Number of Organizations', 'Number of Interests',
                    'Number of Activities']
    
    for col in numeric_cols:
        if col in df.columns:
            features[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Photo column (Yes/No -> 1/0)
    if 'Photo' in df.columns:
        features['Photo'] = df['Photo'].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}).fillna(0)
    
    # Extract length-based features from text columns
    text_cols = ['Intro', 'Full Name', 'Workplace', 'Location', 'About']
    for col in text_cols:
        if col in df.columns:
            features[f'{col}_length'] = df[col].astype(str).str.len().fillna(0)
    
    # Count items in dictionary-like columns
    dict_cols = ['Experiences', 'Educations', 'Licenses', 'Volunteering', 'Skills',
                 'Recommendations', 'Projects', 'Publications', 'Courses', 'Honors',
                 'Scores', 'Languages', 'Organizations', 'Interests', 'Activities']
    
    for col in dict_cols:
        if col in df.columns:
            def count_dict_items(x):
                try:
                    if pd.isna(x) or x == '{}' or x == '':
                        return 0
                    # Try to parse as dict
                    if isinstance(x, str):
                        d = ast.literal_eval(x)
                        if isinstance(d, dict):
                            return len(d)
                    return 0
                except:
                    return 0
            features[f'{col}_count'] = df[col].apply(count_dict_items)
    
    feature_df = pd.DataFrame(features)
    feature_df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(1)
    
    return feature_df


def load_instagram_data(filepath):
    """Load and preprocess Instagram dataset"""
    df = pd.read_csv(filepath)
    
    # Rename columns for consistency
    rename_map = {
        'userFollowerCount': 'Followers',
        'userFollowingCount': 'Following',
        'userBiographyLength': 'Bio_Length',
        'userMediaCount': 'Media_Count',
        'userHasProfilPic': 'Has_Profile_Pic',
        'userIsPrivate': 'Is_Private',
        'usernameDigitCount': 'Username_Digit_Count',
        'usernameLength': 'Username_Length',
        'isFake': 'label'
    }
    
    df = df.rename(columns=rename_map)
    
    # Instagram isFake: 1 = Fake, 0 = Real. We want 1 = Real, 0 = Fake for consistency
    df['label'] = 1 - df['label']
    
    feature_cols = ['Followers', 'Following', 'Bio_Length', 'Media_Count', 
                    'Has_Profile_Pic', 'Is_Private', 'Username_Digit_Count', 'Username_Length']
    
    feature_df = df[feature_cols].copy()
    feature_df['label'] = df['label']
    
    return feature_df


def load_github_data(filepath):
    """Load and preprocess GitHub dataset"""
    df = pd.read_csv(filepath)
    
    # Map label: Human -> 1 (Real), Bot -> 0 (Fake)
    df['label'] = df['label'].map({'Human': 1, 'Bot': 0}).fillna(1)
    
    # Select numeric features
    feature_cols = ['Number of followers', 'Number of following', 'tfidf_similarity',
                    'Number of Activity', 'Number of Issue', 'Number of Pull Request',
                    'Number of Repository', 'Number of Commit', 'Number of Active day',
                    'Periodicity of Activities', 'Number of Connection Account', 'Median Response Time']
    
    # Also add some derived features
    feature_df = df[feature_cols].copy()
    
    # Add binary features from login, name, email, bio, tag (all seem to be 0/1)
    binary_cols = ['login', 'name', 'email', 'bio', 'tag']
    for col in binary_cols:
        if col in df.columns:
            feature_df[f'has_{col}'] = df[col]
    
    feature_df['label'] = df['label']
    
    return feature_df


def align_features(linkedin_df, instagram_df, github_df):
    """Align features across all three datasets"""
    # Get union of all feature columns (excluding 'label')
    all_features = set()
    for df in [linkedin_df, instagram_df, github_df]:
        all_features.update([c for c in df.columns if c != 'label'])
    
    all_features = sorted(list(all_features))
    
    # Align each dataframe to have all features, filling missing with 0
    aligned_dfs = []
    for df, platform_name in [(linkedin_df, 'LinkedIn'), (instagram_df, 'Instagram'), (github_df, 'GitHub')]:
        aligned_df = pd.DataFrame(0, index=df.index, columns=all_features)
        
        for col in df.columns:
            if col != 'label':
                aligned_df[col] = df[col]
        
        aligned_df['label'] = df['label']
        aligned_dfs.append(aligned_df)
    
    return aligned_dfs, all_features


def prepare_training_data(linkedin_df, instagram_df, github_df, all_features):
    """Prepare combined training data with platform IDs"""
    # Add platform_id to each dataset
    linkedin_df['platform_id'] = 0
    instagram_df['platform_id'] = 1
    github_df['platform_id'] = 2
    
    # Combine all datasets
    combined_df = pd.concat([linkedin_df, instagram_df, github_df], ignore_index=True)
    
    # Separate features and labels
    feature_cols = [c for c in all_features] + ['platform_id']
    X = combined_df[feature_cols].values
    y = combined_df['label'].values
    
    return X, y, combined_df


def train_model(X_train, y_train):
    """Train XGBoost model with SMOTE for handling class imbalance"""
    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"After SMOTE - Training samples: {len(y_balanced)}, Class distribution: {np.bincount(y_balanced.astype(int))}")
    
    # Train XGBoost classifier
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='binary:logistic',
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_balanced, y_balanced)
    
    return model


def evaluate_model(model, X_test, y_test, platform_ids=None):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print("\n=== Overall Model Performance ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake/Bot', 'Real/Human']))
    
    # Platform-wise evaluation if platform_ids provided
    if platform_ids is not None:
        platform_names = {0: 'LinkedIn', 1: 'Instagram', 2: 'GitHub'}
        print("\n=== Platform-wise Performance ===")
        for pid in [0, 1, 2]:
            mask = platform_ids == pid
            if mask.sum() > 0:
                y_pred_p = model.predict(X_test[mask])
                y_test_p = y_test[mask]
                acc_p = accuracy_score(y_test_p, y_pred_p)
                f1_p = f1_score(y_test_p, y_pred_p)
                print(f"{platform_names[pid]}: Accuracy={acc_p:.4f}, F1={f1_p:.4f}, Samples={mask.sum()}")
    
    return metrics


def save_model(model, feature_list, filepath='fakeguard_model.pkl'):
    """Save trained model and metadata"""
    model_data = {
        'model': model,
        'feature_list': feature_list,
        'platform_map': {0: 'LinkedIn', 1: 'Instagram', 2: 'GitHub'}
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to {filepath}")


def load_model(filepath='fakeguard_model.pkl'):
    """Load trained model"""
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    return model_data


def predict_fake(model_data, account_features, platform='unknown'):
    """
    Predict if an account is fake or real
    
    Parameters:
    -----------
    model_data : dict
        Loaded model data containing model, feature_list, platform_map
    account_features : dict
        Dictionary of feature names and values for the account
    platform : str
        Platform name ('LinkedIn', 'Instagram', 'GitHub')
    
    Returns:
    --------
    dict : Prediction result with label, confidence, and platform
    """
    model = model_data['model']
    feature_list = model_data['feature_list']
    
    # Map platform name to ID
    platform_map_rev = {v: k for k, v in model_data['platform_map'].items()}
    platform_id = platform_map_rev.get(platform, 0)
    
    # Create feature vector
    feature_vector = []
    for feat in feature_list:
        feature_vector.append(account_features.get(feat, 0))
    feature_vector.append(platform_id)  # Add platform_id
    
    X = np.array([feature_vector])
    
    # Predict
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    
    result = {
        'label': 'Real/Human' if prediction == 1 else 'Fake/Bot',
        'prediction': int(prediction),
        'confidence': float(probability[prediction]),
        'probabilities': {
            'Fake/Bot': float(probability[0]),
            'Real/Human': float(probability[1])
        },
        'platform': platform
    }
    
    return result


def main():
    """Main function to run the complete pipeline"""
    print("=" * 60)
    print("FakeGuard AI - Unified Fake Account Detection")
    print("=" * 60)
    
    # Load datasets
    print("\n[1/6] Loading datasets...")
    linkedin_df = load_linkedin_data('LinkedIn_Dataset.csv')
    print(f"  LinkedIn: {len(linkedin_df)} samples, {len(linkedin_df.columns)-1} features")
    
    instagram_df = load_instagram_data('instagram_dataset.csv')
    print(f"  Instagram: {len(instagram_df)} samples, {len(instagram_df.columns)-1} features")
    
    github_df = load_github_data('bothawk_data_Ori.csv')
    print(f"  GitHub: {len(github_df)} samples, {len(github_df.columns)-1} features")
    
    # Align features
    print("\n[2/6] Aligning features across platforms...")
    aligned_dfs, all_features = align_features(linkedin_df, instagram_df, github_df)
    print(f"  Unified feature set: {len(all_features)} features")
    
    # Prepare training data
    print("\n[3/6] Preparing training data with platform IDs...")
    X, y, combined_df = prepare_training_data(aligned_dfs[0], aligned_dfs[1], aligned_dfs[2], all_features)
    print(f"  Total samples: {len(y)}")
    print(f"  Class distribution before SMOTE: {np.bincount(y.astype(int))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Extract platform IDs for test set
    platform_col_idx = len(all_features)
    platform_ids_test = X_test[:, platform_col_idx]
    
    print(f"  Training set: {len(y_train)} samples")
    print(f"  Test set: {len(y_test)} samples")
    
    # Train model
    print("\n[4/6] Training XGBoost model with SMOTE...")
    model = train_model(X_train, y_train)
    
    # Evaluate model
    print("\n[5/6] Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test, platform_ids_test)
    
    # Save model
    print("\n[6/6] Saving model...")
    save_model(model, all_features)
    
    # Example prediction
    print("\n" + "=" * 60)
    print("Example Predictions:")
    print("=" * 60)
    
    model_data = load_model()
    
    # Example LinkedIn account
    linkedin_example = {feat: 0 for feat in all_features}
    linkedin_example.update({
        'Followers': 500,
        'Connections': 250,
        'Photo': 1,
        'Number of Skills': 10,
        'Number of Experiences': 3
    })
    pred_li = predict_fake(model_data, linkedin_example, platform='LinkedIn')
    print(f"\nLinkedIn Account: {pred_li['label']} (Confidence: {pred_li['confidence']:.2%})")
    
    # Example Instagram account
    instagram_example = {feat: 0 for feat in all_features}
    instagram_example.update({
        'Followers': 1000,
        'Following': 500,
        'Has_Profile_Pic': 1,
        'Media_Count': 50
    })
    pred_ig = predict_fake(model_data, instagram_example, platform='Instagram')
    print(f"Instagram Account: {pred_ig['label']} (Confidence: {pred_ig['confidence']:.2%})")
    
    # Example GitHub account
    github_example = {feat: 0 for feat in all_features}
    github_example.update({
        'Number of followers': 100,
        'Number of following': 50,
        'Number of Repository': 20,
        'Number of Commit': 500
    })
    pred_gh = predict_fake(model_data, github_example, platform='GitHub')
    print(f"GitHub Account: {pred_gh['label']} (Confidence: {pred_gh['confidence']:.2%})")
    
    print("\n" + "=" * 60)
    print("FakeGuard AI training complete!")
    print("=" * 60)
    
    return model_data, metrics


if __name__ == '__main__':
    model_data, metrics = main()
