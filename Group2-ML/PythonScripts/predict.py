import joblib
import numpy as np

# Load the saved models
log_reg_model = joblib.load('PythonScripts/log_reg_model.pkl')
svm_model = joblib.load('PythonScripts/svm_model.pkl')
ada_model = joblib.load('PythonScripts/ada_model.pkl')
stacking_model = joblib.load('PythonScripts/stacking_model.pkl')

def predict(model_name, *args):
    input_features = np.array([args])

    if model_name == 'log_reg':
        model = log_reg_model
    elif model_name == 'svm':
        model = svm_model
    elif model_name == 'ada':
        model = ada_model
    elif model_name == 'stacking':
        model = stacking_model
    else:
        raise ValueError(f"Unknown model: {model_name}")

    prediction = model.predict(input_features)
    return "Edible" if prediction[0] == 0 else "Poisonous"
