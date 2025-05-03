import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score,
                            f1_score, roc_curve, auc, precision_recall_curve, average_precision_score,
                            cohen_kappa_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(features, labels, feature_names, params):
    """
    Train a machine learning model for epilepsy detection
    
    Parameters:
    -----------
    features : ndarray
        Features with shape (n_windows, n_features)
    labels : ndarray
        Labels for each window (0 for non-seizure, 1 for seizure)
    feature_names : list
        Names of the features
    params : dict
        Training parameters
    
    Returns:
    --------
    model : object
        Trained machine learning model
    predictions : dict
        Dictionary containing test set predictions
    metrics : dict
        Dictionary containing performance metrics
    feature_importances : dict
        Dictionary containing feature importances
    """
    try:
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=params['test_size'], random_state=42, stratify=labels
        )
        
        logging.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
        logging.info(f"Training set label distribution: {np.bincount(y_train)}")
        logging.info(f"Test set label distribution: {np.bincount(y_test)}")
        
        # Apply class balancing if requested
        if params['class_balance']['apply']:
            X_train_balanced, y_train_balanced = apply_class_balancing(
                X_train, y_train, params['class_balance']['method']
            )
            logging.info(f"After balancing - Training set: {X_train_balanced.shape[0]} samples")
            logging.info(f"After balancing - Label distribution: {np.bincount(y_train_balanced)}")
        else:
            X_train_balanced = X_train
            y_train_balanced = y_train
        
        # Apply feature selection if requested
        if params['feature_selection']['apply']:
            X_train_selected, X_test_selected, selected_feature_names = apply_feature_selection(
                X_train_balanced, X_test, y_train_balanced, feature_names, params['feature_selection']
            )
            logging.info(f"After feature selection - Features: {X_train_selected.shape[1]}")
        else:
            X_train_selected = X_train_balanced
            X_test_selected = X_test
            selected_feature_names = feature_names
        
        # Create and train model
        model = create_model(params['model_type'])
        
        # Apply hyperparameter tuning if requested
        if params['hyperparameter_tuning']:
            model = tune_hyperparameters(
                model, X_train_selected, y_train_balanced, params['model_type'], params['cv_folds']
            )
        
        # Train the model
        model.fit(X_train_selected, y_train_balanced)
        
        # Make predictions on test set
        y_pred = model.predict(X_test_selected)
        
        # Get probability estimates if the model supports it
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
        else:
            # Use decision function if available
            if hasattr(model, 'decision_function'):
                y_pred_proba = model.decision_function(X_test_selected)
                # Scale to [0, 1] range
                y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
            else:
                y_pred_proba = y_pred.astype(float)
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(y_test, y_pred, y_pred_proba)
        
        # Get feature importances
        feature_importances = get_feature_importances(model, selected_feature_names, params['model_type'])
        
        # Store predictions
        predictions = {
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        return model, predictions, metrics, feature_importances
    
    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        return None, None, None, None

def apply_class_balancing(X_train, y_train, method):
    """
    Apply class balancing to the training data
    
    Parameters:
    -----------
    X_train : ndarray
        Training features
    y_train : ndarray
        Training labels
    method : str
        Class balancing method
    
    Returns:
    --------
    X_train_balanced : ndarray
        Balanced training features
    y_train_balanced : ndarray
        Balanced training labels
    """
    try:
        if method == 'SMOTE':
            # Apply SMOTE
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            logging.info("Applied SMOTE for class balancing")
        
        elif method == 'Random Undersampling':
            # Apply random undersampling
            rus = RandomUnderSampler(random_state=42)
            X_train_balanced, y_train_balanced = rus.fit_resample(X_train, y_train)
            logging.info("Applied Random Undersampling for class balancing")
        
        elif method == 'Class Weights':
            # Use class weights in the model (no resampling needed)
            X_train_balanced = X_train
            y_train_balanced = y_train
            logging.info("Using Class Weights for class balancing (will be applied during model training)")
        
        else:
            # No balancing
            X_train_balanced = X_train
            y_train_balanced = y_train
        
        return X_train_balanced, y_train_balanced
    
    except Exception as e:
        logging.error(f"Error during class balancing: {str(e)}")
        return X_train, y_train

def apply_feature_selection(X_train, X_test, y_train, feature_names, params):
    """
    Apply feature selection
    
    Parameters:
    -----------
    X_train : ndarray
        Training features
    X_test : ndarray
        Test features
    y_train : ndarray
        Training labels
    feature_names : list
        Names of the features
    params : dict
        Feature selection parameters
    
    Returns:
    --------
    X_train_selected : ndarray
        Selected training features
    X_test_selected : ndarray
        Selected test features
    selected_feature_names : list
        Names of the selected features
    """
    try:
        method = params['method']
        
        if method == 'Recursive Feature Elimination':
            # Get number of features to select
            n_features = params['n_features']
            
            # Create a baseline model for RFE
            base_model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Apply RFE
            rfe = RFE(estimator=base_model, n_features_to_select=n_features, step=1)
            rfe.fit(X_train, y_train)
            
            # Get selected features
            selected_indices = np.where(rfe.support_)[0]
            X_train_selected = X_train[:, selected_indices]
            X_test_selected = X_test[:, selected_indices]
            
            # Get selected feature names
            selected_feature_names = [feature_names[i] for i in selected_indices]
            
            logging.info(f"Applied RFE and selected {len(selected_feature_names)} features")
        
        elif method == 'Select K Best':
            # Get number of features to select
            n_features = params['n_features']
            
            # Apply SelectKBest
            selector = SelectKBest(score_func=f_classif, k=n_features)
            selector.fit(X_train, y_train)
            
            # Get selected features
            selected_indices = np.where(selector.get_support())[0]
            X_train_selected = X_train[:, selected_indices]
            X_test_selected = X_test[:, selected_indices]
            
            # Get selected feature names
            selected_feature_names = [feature_names[i] for i in selected_indices]
            
            logging.info(f"Applied SelectKBest and selected {len(selected_feature_names)} features")
        
        elif method == 'Principal Component Analysis':
            # Get variance to retain
            variance_retained = params['variance_retained']
            
            # Apply PCA
            pca = PCA(n_components=variance_retained, random_state=42)
            X_train_selected = pca.fit_transform(X_train)
            X_test_selected = pca.transform(X_test)
            
            # Create new feature names for PCA components
            selected_feature_names = [f"PCA_Component_{i+1}" for i in range(X_train_selected.shape[1])]
            
            logging.info(f"Applied PCA and reduced to {len(selected_feature_names)} components")
            
            # For interpretability, also calculate feature importance in the context of PCA
            # by examining the loading of each original feature on the principal components
            if feature_names is not None:
                logging.info(f"Top features contributing to the first 3 principal components:")
                for i in range(min(3, pca.components_.shape[0])):
                    loadings = pca.components_[i]
                    sorted_indices = np.argsort(np.abs(loadings))[::-1][:5]  # Top 5 features for each component
                    top_features = [(feature_names[idx], loadings[idx]) for idx in sorted_indices]
                    logging.info(f"PC{i+1}: {top_features}")
        
        else:
            # No feature selection
            X_train_selected = X_train
            X_test_selected = X_test
            selected_feature_names = feature_names
        
        return X_train_selected, X_test_selected, selected_feature_names
    
    except Exception as e:
        logging.error(f"Error during feature selection: {str(e)}")
        return X_train, X_test, feature_names

def create_model(model_type):
    """
    Create a machine learning model based on the specified type
    
    Parameters:
    -----------
    model_type : str
        Type of model to create
    
    Returns:
    --------
    model : object
        Machine learning model
    """
    if model_type == 'Random Forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
    
    elif model_type == 'Support Vector Machine':
        model = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            probability=True,
            random_state=42
        )
    
    elif model_type == 'XGBoost':
        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    
    elif model_type == 'Logistic Regression':
        model = LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='liblinear',
            random_state=42
        )
    
    elif model_type == 'Neural Network':
        model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
    
    else:
        # Default to Random Forest
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
    
    return model

def tune_hyperparameters(model, X_train, y_train, model_type, cv_folds):
    """
    Tune hyperparameters of the model using grid search cross-validation
    
    Parameters:
    -----------
    model : object
        Machine learning model
    X_train : ndarray
        Training features
    y_train : ndarray
        Training labels
    model_type : str
        Type of model
    cv_folds : int
        Number of cross-validation folds
    
    Returns:
    --------
    best_model : object
        Model with the best hyperparameters
    """
    # Define parameter grids for different model types
    if model_type == 'Random Forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    
    elif model_type == 'Support Vector Machine':
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.1, 0.01]
        }
    
    elif model_type == 'XGBoost':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    
    elif model_type == 'Logistic Regression':
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
    
    elif model_type == 'Neural Network':
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
    
    else:
        # Default parameter grid for Random Forest
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
        scoring='f1',
        n_jobs=-1
    )
    
    # Fit grid search
    logging.info(f"Performing hyperparameter tuning for {model_type} using grid search")
    grid_search.fit(X_train, y_train)
    
    # Get best parameters
    best_params = grid_search.best_params_
    logging.info(f"Best parameters: {best_params}")
    
    # Return best model
    return grid_search.best_estimator_

def calculate_performance_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculate performance metrics for model evaluation
    
    Parameters:
    -----------
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
    y_pred_proba : ndarray
        Predicted probabilities
    
    Returns:
    --------
    metrics : dict
        Dictionary containing performance metrics
    """
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)  # Same as recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate PPV and NPV
    ppv = precision  # Same as precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # Calculate Cohen's Kappa
    cohen_kappa = cohen_kappa_score(y_true, y_pred)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Calculate precision-recall curve and average precision
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
    average_precision = average_precision_score(y_true, y_pred_proba)
    
    # Store metrics in a dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_score': f1,
        'ppv': ppv,
        'npv': npv,
        'cohen_kappa': cohen_kappa,
        'roc_auc': roc_auc,
        'average_precision': average_precision,
        'fpr': fpr,
        'tpr': tpr,
        'precision_curve': precision_curve,
        'recall_curve': recall_curve,
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    return metrics

def get_feature_importances(model, feature_names, model_type):
    """
    Get feature importances from the trained model
    
    Parameters:
    -----------
    model : object
        Trained machine learning model
    feature_names : list
        Names of the features
    model_type : str
        Type of model
    
    Returns:
    --------
    feature_importances : dict
        Dictionary containing feature importances
    """
    try:
        # Initialize empty feature importances
        importances = None
        
        # Extract feature importances based on model type
        if model_type == 'Random Forest':
            importances = model.feature_importances_
        
        elif model_type == 'Support Vector Machine':
            # For linear SVM
            if model.kernel == 'linear':
                importances = np.abs(model.coef_[0])
            else:
                # For non-linear SVM, no direct feature importances
                pass
        
        elif model_type == 'XGBoost':
            importances = model.feature_importances_
        
        elif model_type == 'Logistic Regression':
            importances = np.abs(model.coef_[0])
        
        elif model_type == 'Neural Network':
            # For neural networks, no direct feature importances
            pass
        
        # Create feature importances dictionary
        if importances is not None and feature_names is not None:
            # Normalize importances
            importances = importances / np.sum(importances) if np.sum(importances) > 0 else importances
            
            # Sort features by importance
            sorted_indices = np.argsort(importances)[::-1]
            sorted_importances = importances[sorted_indices]
            sorted_names = [feature_names[i] for i in sorted_indices]
            
            feature_importances = {
                'names': sorted_names,
                'values': sorted_importances
            }
        else:
            feature_importances = None
        
        return feature_importances
    
    except Exception as e:
        logging.error(f"Error getting feature importances: {str(e)}")
        return None

def evaluate_model(model, features, labels):
    """
    Evaluate a trained model on new data
    
    Parameters:
    -----------
    model : object
        Trained machine learning model
    features : ndarray
        Features to evaluate on
    labels : ndarray
        True labels
    
    Returns:
    --------
    predictions : dict
        Dictionary containing predictions
    metrics : dict
        Dictionary containing performance metrics
    """
    try:
        # Make predictions
        y_pred = model.predict(features)
        
        # Get probability estimates if the model supports it
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(features)[:, 1]
        else:
            # Use decision function if available
            if hasattr(model, 'decision_function'):
                y_pred_proba = model.decision_function(features)
                # Scale to [0, 1] range
                y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
            else:
                y_pred_proba = y_pred.astype(float)
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(labels, y_pred, y_pred_proba)
        
        # Store predictions
        predictions = {
            'y_true': labels,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        return predictions, metrics
    
    except Exception as e:
        logging.error(f"Error during model evaluation: {str(e)}")
        return None, None
