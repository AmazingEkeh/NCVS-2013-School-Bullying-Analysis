import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import roc_curve
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

def load_and_preprocess_data(filepath, target_column, columns_to_drop=None):
    df = pd.read_csv(filepath)

    if columns_to_drop:
        df.drop(columns=columns_to_drop, inplace=True)

    null_values = df.isnull()
    if null_values.any().any():
        df.dropna(axis=0, inplace=True)
    else:
        print("There aren't any null values in this dataset.")

    # Keeping it as Series and retaining the name attribute
    target_variable = df[target_column].copy()
    df.drop(columns=[target_column], inplace=True)

    scaler = StandardScaler()
    data_standardized = pd.DataFrame(scaler.fit_transform(
        df), columns=df.columns)  # Keeping as DataFrame

    return data_standardized, target_variable

# Function to train a classification model (Logistic Regression or SVM) 
def train_classification_model(data, target_variable, top_n_features, model_type='LogisticRegression'):
    # Convert data back to DataFrame
    data = pd.DataFrame(data, columns=data.columns)

    # Select top N features
    top_features = data.corrwith(
        target_variable).abs().nlargest(top_n_features).index
    data_filtered = data[top_features]

    # Applying SMOTE to address class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(
        data_filtered, target_variable)

    # Splitting the resampled data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Initialize and train the classification model
    if model_type == 'LogisticRegression':
        clf = LogisticRegression(
            max_iter=1000, random_state=42, class_weight='balanced')
    elif model_type == 'SVM':
        clf = SVC(C=10, kernel='linear', class_weight='balanced',
                  random_state=42)  # Define and initialize the SVC model

    clf.fit(X_train, y_train)  # Fit the model

    # Making predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculating the accuracy and displaying the classification report
    acc = accuracy_score(y_test, y_pred)
    print(
        f'\nAccuracy with top {top_n_features} features ({model_type}): {acc:.2f}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    return clf

# Function to build and train a DNN model
def build_and_train_dnn_model(data, target_variable):
    # Splitting the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, target_variable, test_size=0.2, random_state=42)

    # Applying SMOTE to the training data (not to the validation/test data)
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Build the DNN model
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        # Binary classification, hence we use sigmoid activation function
        layers.Dense(1, activation='sigmoid'),
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Early stopping callback
    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                        validation_split=0.1, callbacks=[early_stopping])

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test set accuracy: {accuracy:.2f}')

    # Generate a classification report
    y_pred_prob = model.predict(X_test)

    # Get the false positive rate, true positive rate, and all thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    # Get the best threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f'Optimal threshold: {optimal_threshold}')

    y_pred = (y_pred_prob > optimal_threshold).astype("int32")
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    return model, history


def train_xgboost_model(data, target_variable, top_n_features):
    # Convert data back to DataFrame
    data = pd.DataFrame(data, columns=data.columns)

    # Select top N features
    top_features = data.corrwith(
        target_variable).abs().nlargest(top_n_features).index
    data_filtered = data[top_features]

    # Applying SMOTE to address class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(
        data_filtered, target_variable)

    # Splitting the resampled data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Initialize and train the XGBoost model
    xgb_classifier = xgb.XGBClassifier(
        objective="binary:logistic", random_state=42, scale_pos_weight=1)

    xgb_classifier.fit(X_train, y_train)  # Fit the model

    # Making predictions on the test set
    y_pred = xgb_classifier.predict(X_test)

    # Calculating the accuracy and displaying the classification report
    acc = accuracy_score(y_test, y_pred)
    print(
        f'\nAccuracy with top {top_n_features} features (XGBoost): {acc:.2f}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    return xgb_classifier

# Function to train a Random Forest model
def train_random_forest_model(data, target_variable, top_n_features, n_estimators=100, max_depth=None):
    # Convert data back to DataFrame
    data = pd.DataFrame(data, columns=data.columns)

    # Select top N features
    top_features = data.corrwith(
        target_variable).abs().nlargest(top_n_features).index
    data_filtered = data[top_features]

    # Applying SMOTE to address class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(
        data_filtered, target_variable)

    # Splitting the resampled data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest model
    rf_classifier = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42, class_weight='balanced')

    rf_classifier.fit(X_train, y_train)  # Fit the model

    # Making predictions on the test set
    y_pred = rf_classifier.predict(X_test)

    # Calculating the accuracy and displaying the classification report
    acc = accuracy_score(y_test, y_pred)
    print(
        f'\nAccuracy with top {top_n_features} features (Random Forest): {acc:.2f}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    return rf_classifier


# Function to train a Naive Bayes model
def train_naive_bayes_model(data, target_variable, top_n_features):
    # Convert data back to DataFrame
    data = pd.DataFrame(data, columns=data.columns)

    # Select top N features
    top_features = data.corrwith(
        target_variable).abs().nlargest(top_n_features).index
    data_filtered = data[top_features]

    # Applying SMOTE to address class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(
        data_filtered, target_variable)

    # Splitting the resampled data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Initialize and train the Naive Bayes model (Gaussian Naive Bayes)
    nb_classifier = GaussianNB()

    nb_classifier.fit(X_train, y_train)  # Fit the model

    # Making predictions on the test set
    y_pred = nb_classifier.predict(X_test)

    # Calculating the accuracy and displaying the classification report
    acc = accuracy_score(y_test, y_pred)
    print(
        f'\nAccuracy with top {top_n_features} features (Naive Bayes): {acc:.2f}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    return nb_classifier

# Function to train a K Nearest Neighbor model
def train_KNN_model(data, target_variable, top_n_features):
    # Convert data back to DataFrame
    data = pd.DataFrame(data, columns=data.columns)

    # Select top N features
    top_features = data.corrwith(
        target_variable).abs().nlargest(top_n_features).index
    data_filtered = data[top_features]

    # Applying SMOTE to address class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(
        data_filtered, target_variable)

    # Splitting the resampled data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Initialize and train the K Nearest Neighbor model
    knn_classifier = KNeighborsClassifier()

    knn_classifier.fit(X_train, y_train)  # Fit the model

    # Making predictions on the test set
    y_pred = knn_classifier.predict(X_test)

    # Calculating the accuracy and displaying the classification report
    acc = accuracy_score(y_test, y_pred)
    print(
        f'\nAccuracy with top {top_n_features} features (K Nearest Neighbor): {acc:.2f}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    return knn_classifier


# Usage
data_filepath = '/Users/user/Desktop/Fall Semester/CS699 A1 Data Mining - Jae Lee/Project/project_dataset.csv'
train_df = pd.read_csv(data_filepath)
data_standardized, target_variable = load_and_preprocess_data(
    data_filepath, target_column='o_bullied', columns_to_drop=None)

# Train models with top N variables for Logistic Regression, SVM, XGBoost, Random Forest, Naive Bayes, and K-Nearest Neighbor
top_features_list = [50, 60, 70, 80, 90, 100,
                     110, 120, 130, 140, 150, 160, 170, 180]

for n_features in top_features_list:
    # Select top N features
    top_features = data_standardized.corrwith(
        target_variable).abs().nlargest(n_features).index
    data_filtered = data_standardized[top_features]

    # Applying SMOTE to address class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(
        data_filtered, target_variable)

    # Splitting the resampled data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Define hyperparameter grids for GridSearch
    if n_features <= 100:
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
        }
        grid_model = GridSearchCV(SVC(
            class_weight='balanced', random_state=42), param_grid, cv=3, n_jobs=-1, scoring='recall')
        grid_model.fit(X_train, y_train)
        best_svm = grid_model.best_estimator_
        print(f'Best SVM Model with {n_features} features: {best_svm}')
    else:
        best_svm = None

    # Train and evaluate Logistic Regression
    train_classification_model(
        data_filtered, target_variable, n_features, model_type='LogisticRegression')

    # Train and evaluate SVM
    if best_svm:
        print(f'SVM for {n_features} features (GridSearch):')
        best_svm.fit(X_train, y_train)  # Fit the best SVM model directly
        train_classification_model(
            data_filtered, target_variable, n_features, model_type='SVM')

    # Using RandomizedSearchCV for Random Forest
    param_dist_rf = {
        'n_estimators': randint(100, 300),
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    randomized_search_rf = RandomizedSearchCV(RandomForestClassifier(random_state=42),
                                              param_distributions=param_dist_rf,
                                              n_iter=20, cv=3, n_jobs=-1, scoring='recall')
    randomized_search_rf.fit(X_train, y_train)
    best_rf = randomized_search_rf.best_estimator_
    print(f'Best Random Forest Model with {n_features} features: {best_rf}')
    # Train and evaluate Random Forest
    train_random_forest_model(data_filtered, target_variable, n_features)

    # Train and evaluate Naive Bayes
    train_naive_bayes_model(data_filtered, target_variable, n_features)

    # Train and evaluate K Nearest Neighbors
    train_KNN_model(data_filtered, target_variable, n_features)

    # Implement GridSearch for XGBoost
    param_grid_xgb = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    grid_search_xgb = GridSearchCV(xgb.XGBClassifier(objective="binary:logistic", random_state=42, scale_pos_weight=1),
                                   param_grid_xgb, cv=3, n_jobs=-1, scoring='recall')
    grid_search_xgb.fit(X_train, y_train)
    best_xgb = grid_search_xgb.best_estimator_
    print(f'Best XGBoost Model with {n_features} features: {best_xgb}')
    # Train and evaluate XGBoost
    train_xgboost_model(data_filtered, target_variable, n_features)




