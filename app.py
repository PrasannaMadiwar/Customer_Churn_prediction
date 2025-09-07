import numpy as np
from flask import Flask, url_for, redirect, render_template, request
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt 
import shap
from sklearn.pipeline import Pipeline

app = Flask(__name__)

model = pickle.load(open("Churn_pred_model","rb")) 
categorical_features = ["Geography", "Card Type"] 
features = [ "CreditScore",
    "Geography",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Exited",
    "Complain",
    "Satisfaction Score",
    "Card Type",
    "Point Earned",
    "Gender_Male"]

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict",methods=['POST'])
def predict():

    input_value = []
    missing_fields = []

    for i in features:
        value = request.form.get(i)
        if value is None or value == '':
                missing_fields.append(i)
        else:
            try:
                if i in categorical_features:
                    input_value.append(value)
                else :
                    input_value.append(float(value))  
            except ValueError:
                return f"Error: Invalid numeric value for field {i}"
            
    if missing_fields:
            return f"Error: Missing values for fields: {', '.join(missing_fields)}"        


    data = np.array(input_value).reshape(1, -1)        
    predict = model.predict(data)
    prob = model.predict_proba(data)

    explainer = shap.TreeExplainer(model.steps[-1][1])
    preprocessor = Pipeline(model.steps[:-1])
    data_pro = preprocessor.transform(data)
    sha_values = explainer.shap_values(data_pro)
    shap_img_path = os.path.join("static","shap.png")

    plt.figure(figsize=(14,4))
    shap.force_plot(explainer.expected_value,sha_values[0],data_pro[0],feature_names=features,matplotlib=True,show=False)
    plt.tight_layout()
    plt.savefig(shap_img_path,bbox_inches='tight',dpi=150,facecolor="white")
    plt.close()

    feature_values = np.abs(sha_values[0])

    return render_template("dashboard.html",
                           prediction = predict[0],
                           probability = prob[0],
                           shap_img = "shap.png",
                           feature_names = features,
                           feature_values = feature_values,
                           Datetime = datetime.now() 
                           )

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT",5000)))