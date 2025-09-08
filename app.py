import numpy as np
import pandas as pd
from flask import Flask, url_for, redirect, render_template, request, jsonify
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt 
import shap
from sklearn.pipeline import Pipeline

app = Flask(__name__)

model = pickle.load(open("Churn_pred","rb")) 
categorical_features = ["Geography", "Card Type"] 
features = ["CreditScore",
    "Geography",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Complain",
    "Satisfaction Score",
    "Card Type",
    "Point Earned",
    "Gender_Male"]

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/api/predict",methods=['POST'])
def predcit_api():
    data = request.get_json()
    df = pd.DataFrame([data])

    pred = model.predict(df)
    prob = model.predict_proba(df)

    explainer = shap.TreeExplainer(model.steps[-1][1])
    preprocessor = Pipeline(model.steps[:-1])
    data_pro = preprocessor.transform(df)
    shap_values = explainer.shap_values(data_pro)

    feature_importance = []
    for i , feature_name in enumerate(features):
         feature_importance.append(
              {
                   "feature":feature_name,
                   "value":data[feature_name],
                   "shap_value":float(shap_values[0][i]),
                   "Contribution":"Positive" if shap_values[0][i] > 0 else "Neagative"
              }
         ) 
    feature_importance.sort(key=lambda x:abs(x["shap_value"]),reverse=True)
    response = {
        "Prediction": {
            "result": int(pred[0]) if hasattr(pred[0], "item") else pred[0],  # int/float
            "prob": prob[0].tolist() if hasattr(prob[0], "tolist") else prob[0]  # list
        },
        "Feature_analysis": {
            "top_5": feature_importance[:5],
            "All_features": feature_importance
        },
        "Time_Stamp": str(datetime.now())
    }
    return jsonify(response), 200

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


    data = pd.DataFrame([input_value], columns=features)        
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
    zipped_features = list(zip(features, feature_values))

    return render_template("dashboard.html",
                           prediction = predict[0],
                           probability = prob[0],
                           shap_img = "shap.png",
                           zipped_features = zipped_features,
                           Datetime = datetime.now() 
                           )

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT",5000)))