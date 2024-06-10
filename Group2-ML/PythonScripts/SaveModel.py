import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
file_path = 'mushrooms.csv'
mushrooms_df = pd.read_csv(file_path)

# Display basic information and statistics of the dataset
print(mushrooms_df.info())
print(mushrooms_df.describe(include='all'))

# Plot class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='class', data=mushrooms_df)
plt.title('Class Distribution')
plt.show()

# Encode categorical variables
label_encoders = {}
for column in mushrooms_df.columns:
    le = LabelEncoder()
    mushrooms_df[column] = le.fit_transform(mushrooms_df[column])
    label_encoders[column] = le

# Separate features and target
X = mushrooms_df.drop('class', axis=1)
y = mushrooms_df['class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize variables to track the best model
best_model = None
best_accuracy = 0

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
log_reg_predictions = log_reg.predict(X_test)
log_reg_accuracy = accuracy_score(y_test, log_reg_predictions)
print("Logistic Regression Accuracy:", log_reg_accuracy)
print(classification_report(y_test, log_reg_predictions))

# SVM
svm = SVC()
svm.fit(X_train, y_train)
svm_predictions = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print("SVM Accuracy:", svm_accuracy)
print(classification_report(y_test, svm_predictions))

# AdaBoost
ada = AdaBoostClassifier()
ada.fit(X_train, y_train)
ada_predictions = ada.predict(X_test)
ada_accuracy = accuracy_score(y_test, ada_predictions)
print("AdaBoost Accuracy:", ada_accuracy)
print(classification_report(y_test, ada_predictions))

# Cross-validation for AdaBoost
ada_scores = cross_val_score(AdaBoostClassifier(), X, y, cv=5)
print(f"AdaBoost Cross-Validation Scores: {ada_scores}")
print(f"Mean CV Accuracy for AdaBoost: {ada_scores.mean()}")

# Confusion Matrix for AdaBoost
ada_cm = confusion_matrix(y_test, ada_predictions)
print(f"Confusion Matrix for AdaBoost:\n{ada_cm}")

# Stacking Ensemble
base_models = [
    ('log_reg', LogisticRegression(max_iter=1000)),
    ('svm', SVC(probability=True)),
    ('ada', AdaBoostClassifier())
]
stacking = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression())
stacking.fit(X_train, y_train)
stacking_predictions = stacking.predict(X_test)
stacking_accuracy = accuracy_score(y_test, stacking_predictions)
print("Stacking Ensemble Accuracy:", stacking_accuracy)
print(classification_report(y_test, stacking_predictions))

# FCNN (Fully Connected Neural Network)
fcnn_model = Sequential()
fcnn_model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
fcnn_model.add(Dense(32, activation='relu'))
fcnn_model.add(Dense(1, activation='sigmoid'))

fcnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = fcnn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

loss, fcnn_accuracy = fcnn_model.evaluate(X_test, y_test)
print("FCNN Accuracy:", fcnn_accuracy)

# Find and save the best model
best_model = max(
    [('Logistic Regression', log_reg, log_reg_accuracy),
     ('SVM', svm, svm_accuracy),
     ('AdaBoost', ada, ada_accuracy),
     ('Stacking Classifier', stacking, stacking_accuracy),
     ('FCNN', fcnn_model, fcnn_accuracy)],
    key=lambda x: x[2]
)

model_name, model_instance, model_accuracy = best_model
print(f"Best model is {model_name} with accuracy {model_accuracy}")

if model_name == 'FCNN':
    model_instance.save('best_model.keras')
else:
    joblib.dump(model_instance, 'best_model.pkl')

# Plotting the accuracies of different models
accuracies = {
    'Logistic Regression': log_reg_accuracy,
    'SVM': svm_accuracy,
    'AdaBoost': ada_accuracy,
    'Stacking Classifier': stacking_accuracy,
    'FCNN': fcnn_accuracy
}

plt.figure(figsize=(10, 6))
plt.bar(accuracies.keys(), accuracies.values(), color='skyblue')
plt.ylabel('Accuracy')
plt.title('Model Accuracies')
plt.show()

# Plotting the loss and accuracy over epochs for the FCNN
if 'FCNN' in accuracies.keys():
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.show()
