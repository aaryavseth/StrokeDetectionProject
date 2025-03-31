import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, r2_score
)
from sklearn.pipeline import Pipeline
from flask import Flask, request, jsonify

# --------------------------- Load and Preprocess Data ---------------------------
def load_and_preprocess_data(filepath):
    print("📥 Loading dataset...")
    df = pd.read_csv(filepath)
    df.fillna(df.median(numeric_only=True), inplace=True)
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop(["At Risk (Binary)", "Stroke Risk (%)"], axis=1)
    y_class = df["At Risk (Binary)"]
    y_reg = df["Stroke Risk (%)"]

    print(f"✅ Loaded {len(df)} rows and {X.shape[1]} features.")
    return X, y_class, y_reg

# --------------------------- Train Models ---------------------------
def train_models(X, y_class, y_reg):
    print("🔄 Splitting and scaling data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_reg_scaled = scaler.transform(X_train_reg)
    X_test_reg_scaled = scaler.transform(X_test_reg)

    print("🧠 Training classifier with hyperparameter tuning...")
    clf_grid = GridSearchCV(RandomForestClassifier(random_state=42), {
        'n_estimators': [100, 150],
        'max_depth': [None, 10, 20],
    }, cv=3, scoring='accuracy')
    clf_grid.fit(X_train_scaled, y_train)
    clf = clf_grid.best_estimator_

    print("🧠 Training regressor...")
    reg = RandomForestRegressor(n_estimators=150, random_state=42)
    reg.fit(X_train_reg_scaled, y_train_reg)

    # Save models and scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/stroke_risk_classifier.pkl")
    joblib.dump(reg, "models/stroke_risk_regressor.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(X.columns.tolist(), "models/feature_names.pkl")

    print("✅ Models and scaler saved.")
    return clf, reg, scaler, X_test_scaled, y_test, X_test_reg_scaled, y_test_reg, X.columns.tolist()

# --------------------------- Evaluate ---------------------------
def evaluate_models(clf, reg, X_test, y_test, X_test_reg, y_test_reg):
    print("📊 Evaluation Results")

    y_pred = clf.predict(X_test)
    print("\n🟩 Classification Metrics:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    y_pred_reg = reg.predict(X_test_reg)
    print("\n📈 Regression Metrics:")
    print("MAE:", mean_absolute_error(y_test_reg, y_pred_reg))
    print("R2 Score:", r2_score(y_test_reg, y_pred_reg))

# --------------------------- Visualize Feature Importance ---------------------------
def plot_feature_importance(model, feature_names, title, filename):
    importance = model.feature_importances_
    sorted_idx = importance.argsort()[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh([feature_names[i] for i in sorted_idx], importance[sorted_idx])
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"📸 Saved feature importance: {filename}")

# --------------------------- Flask API ---------------------------
def create_flask_app(clf, reg, scaler, feature_names):
    app = Flask(__name__)

    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.json
        input_data = [data[feature] for feature in feature_names]
        input_scaled = scaler.transform([input_data])
        at_risk = clf.predict(input_scaled)[0]
        stroke_risk = reg.predict(input_scaled)[0]
        return jsonify({
            "At Risk": int(at_risk),
            "Stroke Risk (%)": float(stroke_risk)
        })

    return app

# --------------------------- Entry Point ---------------------------
if __name__ == "__main__":
    filepath = "stroke_risk_dataset.csv"  # 🔁 Update if needed
    X, y_class, y_reg = load_and_preprocess_data(filepath)
    clf, reg, scaler, X_test, y_test, X_test_reg, y_test_reg, feat_names = train_models(X, y_class, y_reg)
    evaluate_models(clf, reg, X_test, y_test, X_test_reg, y_test_reg)
    plot_feature_importance(clf, feat_names, "Feature Importance (Classifier)", "classifier_importance.png")
    plot_feature_importance(reg, feat_names, "Feature Importance (Regressor)", "regressor_importance.png")

    # Optional: run API
    # app = create_flask_app(clf, reg, scaler, feat_names)
    # app.run(debug=True)
