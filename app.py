from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
from privacy_core import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        file = request.files["file"]

        if file.filename == "":
            return "No file selected."

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # 1️⃣ LOAD DATASET
        if filepath.endswith(".csv"):
            real_df = pd.read_csv(filepath)
        elif filepath.endswith(".xlsx"):
            real_df = pd.read_excel(filepath)
        else:
            return "Only CSV or Excel files are supported."

        # 2️⃣ KEEP ONLY NUMERIC COLUMNS
        real_df = real_df.select_dtypes(include=[np.number])

        if real_df.shape[1] == 0:
            return "Dataset must contain at least one numeric column."

        # 3️⃣ CLEAN DATA
        real_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        real_df = real_df.fillna(real_df.mean())

        if real_df.shape[0] < 10:
            return "Dataset too small for analysis."

        # 4️⃣ GENERATE SYNTHETIC DATA
        noise_levels = [0.01, 0.05, 0.1, 0.2, 0.3]
        similarities = []
        scores = []

        for noise in noise_levels:
            synthetic_df = real_df + np.random.normal(0, noise, real_df.shape)

            similarity = statistical_similarity(real_df, synthetic_df)
            reid = reidentification_risk(real_df, synthetic_df)
            labels = np.random.randint(0, 2, size=len(real_df))
            membership = membership_inference(real_df, labels)
            score = calculate_privacy_score(similarity, reid, membership)

            similarities.append(similarity)
            scores.append(score)

        # Take final values
        similarity = similarities[-1]
        reid = reidentification_risk(real_df, synthetic_df)
        membership = membership
        score = scores[-1]

        risk_dict = {
            "Re-identification Risk": reid,
            "Membership Inference Risk": membership,
            "Low Statistical Divergence Risk": 100 - similarity
        }

        dominant_risk = max(risk_dict, key=risk_dict.get)

        # Plot
        plt.figure()
        plt.plot(similarities, scores)
        plt.xlabel("Statistical Similarity")
        plt.ylabel("Privacy Score")
        plt.title("Privacy–Utility Tradeoff Curve")
        plt.savefig("static/tradeoff.png")
        plt.close()

        # Status
        if score > 75:
            status = "SAFE TO SHARE ✅"
        elif score > 50:
            status = "MODERATE RISK ⚠"
        else:
            status = "HIGH PRIVACY RISK ❌"

        return render_template("index.html",
                               similarity=round(similarity, 2),
                               reid=round(reid, 2),
                               membership=round(membership, 2),
                               score=round(score, 2),
                               status=status,
                               dominant_risk=dominant_risk)

    # ✅ THIS FIXES YOUR ERROR (GET request)
    return render_template("index.html",
                           similarity=None,
                           reid=None,
                           membership=None,
                           score=None,
                           status=None,
                           dominant_risk=None)

    

if __name__ == "__main__":
    app.run(debug=True)

