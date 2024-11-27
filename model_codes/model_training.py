import logging
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, LSTM, Dropout, SimpleRNN

def train_sklearn_models(preprocessed_data):
    logging.info("Training sklearn models...")
    X_train = preprocessed_data["X_train_balanced_scaled"]
    y_train = preprocessed_data["y_train_balanced"]

    models = {
        "Logistic Regression": GridSearchCV(LogisticRegression(max_iter=5000), {'C': [0.1, 1, 10]}),
        "SVM": GridSearchCV(SVC(probability=True), {'C': [0.1, 1, 10]}),
        "Random Forest": GridSearchCV(RandomForestClassifier(), {'n_estimators': [50, 100]}),
        "XGBoost": GridSearchCV(xgb.XGBClassifier(), {'n_estimators': [50, 100]})
    }

    for name, model in models.items():
        logging.info(f"Training {name}...")
        model.fit(X_train, y_train)

    logging.info("Sklearn models training completed.")
    return models

def train_nn_models(preprocessed_data):
    logging.info("Training neural network models...")
    X_train = preprocessed_data["X_train_balanced_scaled"]
    y_train = preprocessed_data["y_train_categorical"]

    nn_models = {
        "ANN": Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(2, activation='softmax')
        ]),
        "CNN": Sequential([
            Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
            Conv1D(64, 3, activation='relu'),
            Flatten(),
            Dense(2, activation='softmax')
        ]),
        "RNN": Sequential([
            SimpleRNN(64, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)),
            SimpleRNN(32, activation='relu'),
            Dense(2, activation='softmax')
        ]),
        "LSTM": Sequential([
            LSTM(64, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], 1)),
            LSTM(32, activation='tanh'),
            Dense(2, activation='softmax')
        ])
    }

    for name, model in nn_models.items():
        logging.info(f"Training {name}...")
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train[..., None], y_train, epochs=50, batch_size=32, verbose=1)

    logging.info("Neural network models training completed.")
    return nn_models
