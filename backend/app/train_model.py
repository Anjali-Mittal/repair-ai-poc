"""
Trains one XGBoost classifier PER vehicle type (ICE, Scooter, EV) to rank probable
root cause from: fault_code, vehicle_age_months, mileage_km, prior_services.

Vehicle types are trained separately rather than as one shared model + a
vehicle_type feature, because failure physics genuinely differs across them
(fuel/ignition wear vs. battery/BMS degradation) -- one shared model tends to
average across those regimes rather than learn either well. Saves model +
encoders + accuracy per vehicle type for serving and reporting.
"""
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import json

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "repair_history.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_artifacts")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = ["fault_code_enc", "vehicle_age_months", "mileage_km", "prior_services"]


def train_one_vehicle_type(vehicle_type: str, df_vt: pd.DataFrame) -> dict:
    out_dir = os.path.join(MODEL_DIR, vehicle_type)
    os.makedirs(out_dir, exist_ok=True)

    fc_le = LabelEncoder()
    df_vt = df_vt.copy()
    df_vt["fault_code_enc"] = fc_le.fit_transform(df_vt["fault_code"])
    fault_code_fallback = df_vt["fault_code"].value_counts().idxmax()

    target_le = LabelEncoder()
    df_vt["root_cause_enc"] = target_le.fit_transform(df_vt["root_cause"])

    X = df_vt[FEATURE_COLS]
    y = df_vt["root_cause_enc"]

    min_class_count = y.value_counts().min()
    stratify_arg = y if min_class_count >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_arg
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=len(target_le.classes_),
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    report = classification_report(y_test, preds, target_names=target_le.classes_, zero_division=0)
    acc = round(accuracy_score(y_test, preds), 3)
    print(f"\n=== {vehicle_type} (n={len(df_vt)}, accuracy={acc}) ===")
    print(report)

    joblib.dump(model, os.path.join(out_dir, "xgb_model.joblib"))
    joblib.dump(fc_le, os.path.join(out_dir, "fault_code_encoder.joblib"))
    joblib.dump(target_le, os.path.join(out_dir, "root_cause_encoder.joblib"))
    joblib.dump(fault_code_fallback, os.path.join(out_dir, "fault_code_fallback.joblib"))

    return {"vehicle_type": vehicle_type, "n_rows": len(df_vt), "accuracy": acc}


def main():
    df = pd.read_csv(DATA_PATH)

    summary = []
    for vehicle_type, df_vt in df.groupby("vehicle_type"):
        summary.append(train_one_vehicle_type(vehicle_type, df_vt))

    with open(os.path.join(MODEL_DIR, "accuracy_by_vehicle_type.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nPer-vehicle-type accuracy summary:")
    for row in summary:
        print(f"  {row['vehicle_type']:10s} n={row['n_rows']:5d}  accuracy={row['accuracy']}")
    print(f"\nSaved per-type models + encoders under {MODEL_DIR}/<vehicle_type>/")


if __name__ == "__main__":
    main()
