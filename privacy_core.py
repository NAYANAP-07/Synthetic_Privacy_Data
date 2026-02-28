import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import pairwise_distances
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# 1️⃣ Statistical Similarity
def statistical_similarity(real, synthetic):
    scores = []

    for column in real.columns:
        stat, p_value = ks_2samp(real[column], synthetic[column])
        scores.append(p_value)

    return np.mean(scores) * 100  # percentage


# 2️⃣ Improved Re-identification Risk
def reidentification_risk(real, synthetic):
    distances = pairwise_distances(synthetic, real)
    min_distances = distances.min(axis=1)

    # dynamic threshold (5% closest matches)
    threshold = np.percentile(min_distances, 5)

    risk = np.mean(min_distances <= threshold)

    return risk * 100


# 3️⃣ Improved Membership Inference
def membership_inference(real, labels):

    X_train, X_test, y_train, y_test = train_test_split(
        real, labels, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    train_conf = np.mean(np.max(model.predict_proba(X_train), axis=1))
    test_conf = np.mean(np.max(model.predict_proba(X_test), axis=1))

    gap = abs(train_conf - test_conf)

    return gap * 100


# 4️⃣ Final Privacy Score
def calculate_privacy_score(similarity, reid_risk, membership_risk):

    risk_total = (
        (reid_risk * 0.5)
        + (membership_risk * 0.3)
        + ((100 - similarity) * 0.2)
    )

    score = 100 - risk_total

    return round(max(score, 0), 2)
