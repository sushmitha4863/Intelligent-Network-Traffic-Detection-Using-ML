import os
import joblib
import cloudpickle
import json
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from river.ensemble.adaptive_random_forest import AdaptiveRandomForestClassifier
from river.metrics import Accuracy
from sklearn.preprocessing import StandardScaler, LabelEncoder
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

try:
    from scapy.all import sniff
except ImportError:
    print("[WARNING] scapy not installed. Online mode will not work without it.")

# ---------------- CONFIG ----------------
DATASET_PATH = "dataset/master_dataset.csv"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "encoder.pkl"
LGBM_PATH = "lightgbm_model.pkl"
ARF_PATH = "arf_model.pkl"
FEATURES_PATH = "features.json"
CHECKPOINT_PATH = "checkpoint_pred.pkl"

BATCH_SIZE = 1000
DRIFT_CHECK_INTERVAL = 500
DRIFT_THRESHOLD = 0.85
LGBM_EXTRA_TREES_ON_DRIFT = 50  # ADD NEW TREES DURING RETRAIN

# Email defaults
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_USER = "ummehani13005@gmail.com"
SMTP_PASS = "irkr itxq ibfn cwqd"  # paste your app password here

# ---------------- Utilities ----------------
def save_model(obj, path, use_cloudpickle=False):
    with open(path, "wb") as f:
        if use_cloudpickle:
            cloudpickle.dump(obj, f)
        else:
            joblib.dump(obj, f)

def load_model(path, use_cloudpickle=False):
    with open(path, "rb") as f:
        if use_cloudpickle:
            return cloudpickle.load(f)
        else:
            return joblib.load(f)

def send_email(subject: str, body: str, to_addr: str):
    if not SMTP_USER or not SMTP_PASS:
        print("[WARNING] SMTP_USER or SMTP_PASS not set — skipping email.")
        return False
    try:
        msg = MIMEMultipart()
        msg["From"] = SMTP_USER
        msg["To"] = to_addr
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)

        print(f"[INFO] Email sent to {to_addr}: {subject}")
        return True
    except Exception as e:
        print(f"[WARNING] Email send failed: {e}")
        return False

def append_new_samples_to_master(df_new: pd.DataFrame):
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    header = not os.path.exists(DATASET_PATH)
    df_new.to_csv(DATASET_PATH, mode="a", index=False, header=header)

# ---------------- ML helpers ----------------
def preprocess_features(df: pd.DataFrame, scaler, encoder, feature_list):
    # Fill missing features with zeros
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0.0

    # Ensure correct column order
    X_df = df[feature_list].copy()

    # Encode categorical columns
    for col in X_df.columns:
        if X_df[col].dtype == object:
            if encoder:
                try:
                    X_df[col] = encoder.transform(X_df[col].astype(str))
                except Exception:
                    X_df[col] = X_df[col].astype(str).factorize()[0]
            else:
                X_df[col] = X_df[col].astype(str).factorize()[0]

    X_numeric = X_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_scaled = scaler.transform(X_numeric.values)
    return X_scaled, X_numeric.columns.tolist()

def retrain_lgbm_on_drift(lgbm: LGBMClassifier, drift_X: np.ndarray, drift_y: np.ndarray, lgbm_path=LGBM_PATH):
    if len(drift_y) == 0:
        return lgbm
    lgbm.warm_start = True
    lgbm.n_estimators += LGBM_EXTRA_TREES_ON_DRIFT
    lgbm.fit(drift_X, drift_y)
    save_model(lgbm, lgbm_path)
    print(f"[INFO] LGBM retrained on {len(drift_y)} new samples due to drift.")
    return lgbm

# ---------------- Load features ----------------
if not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError(f"[ERROR] Required file missing: {FEATURES_PATH}")

with open(FEATURES_PATH, "r") as f:
    feature_list = json.load(f)

# ---------------- Core processing ----------------
def process_stream(mode: str, user_email: str, input_path: str = None, output_path: str = None,
                   simulate_stream: bool = False, per_detection_email: bool = True):
    # Load models
    scaler = load_model(SCALER_PATH)
    encoder = load_model(ENCODER_PATH)
    lgbm: LGBMClassifier = load_model(LGBM_PATH)
    arf: AdaptiveRandomForestClassifier = load_model(ARF_PATH, use_cloudpickle=True)

    drift_candidates_X, drift_candidates_y = [], []
    total_malicious_detected = 0
    output_rows = []

    # ---------------- Offline mode ----------------
    if mode == "offline":
        if not input_path or not os.path.exists(input_path):
            raise FileNotFoundError(f"[ERROR] Input CSV not found: {input_path}")

        reader = pd.read_csv(input_path, chunksize=BATCH_SIZE)
        sample_counter = 0
        malicious_rows = []
        for chunk in reader:
            y_chunk_raw = chunk["Label"].values if "Label" in chunk.columns else None
            X_chunk_df = chunk.drop(columns=["Label"]) if "Label" in chunk.columns else chunk
            X_chunk, _ = preprocess_features(X_chunk_df, scaler, encoder, feature_list)

            for i_row, x_row in enumerate(X_chunk):
                sample_counter += 1
                x_dict = dict(zip(feature_list, x_row.tolist()))

                try:
                    lgbm_probs = lgbm.predict_proba(x_row.reshape(1, -1))[0]
                    predicted_idx = int(np.argmax(lgbm_probs))
                    predicted_prob = float(np.max(lgbm_probs))
                    THRESHOLD = 0.10

                    # ---------- Safe label decoding ----------
                    if hasattr(encoder, "inverse_transform"):
                        try:
                            predicted_label = encoder.inverse_transform([predicted_idx])[0]
                        except Exception:
                            predicted_label = str(predicted_idx)
                    else:
                        predicted_label = str(predicted_idx)

                    if str(predicted_label).lower() in ["benign", "normal"]:
                        lgbm_label_str = "Benign"
                    else:
                        lgbm_label_str = str(predicted_label)

                    lgbm_prob_1 = predicted_prob

                except Exception as e:
                    print(f"[WARNING] LGBM prediction failed: {e}")
                    lgbm_label_str = "Unknown"
                    lgbm_prob_1 = 0.0

                output_rows.append({
                    "sample_idx": sample_counter,
                    "lgbm_pred": lgbm_label_str,
                    "lgbm_prob_1": lgbm_prob_1,
                    "true_label": y_chunk_raw[i_row] if y_chunk_raw is not None else None
                })

                if lgbm_label_str.lower() not in ["benign", "normal"]:
                    total_malicious_detected += 1
                    row_dict = {c: v for c, v in zip(feature_list, x_row)}
                    row_dict["Label"] = lgbm_label_str
                    append_new_samples_to_master(pd.DataFrame([row_dict]))
                    malicious_rows.append(row_dict)

                    # Add to drift buffer
                    drift_candidates_X.append(x_row)
                    drift_candidates_y.append(predicted_idx)

                # Retrain LGBM if drift buffer reached threshold
                if len(drift_candidates_y) >= DRIFT_CHECK_INTERVAL:
                    drift_X_arr = np.array(drift_candidates_X)
                    drift_y_arr = np.array(drift_candidates_y)
                    lgbm = retrain_lgbm_on_drift(lgbm, drift_X_arr, drift_y_arr)
                    drift_candidates_X.clear()
                    drift_candidates_y.clear()

        if output_path:
            pd.DataFrame(output_rows).to_csv(output_path, index=False)

        # Send email alert
        if user_email:
            if total_malicious_detected > 0:
                body = f"Processed {sample_counter} samples.\n\nDetected malicious traffic:\n"
                for i, r in enumerate(malicious_rows):
                    body += f"\n{i + 1}. {json.dumps(r, indent=2)}"
                send_email("[ALERT] Offline scan complete — Malicious traffic detected", body, to_addr=user_email)
            else:
                send_email("[INFO] Offline scan complete — No malicious traffic found",
                           f"Processed {sample_counter} samples. Everything is normal.",
                           to_addr=user_email)

        print(f"[INFO] Offline processing finished. Total malicious detected: {total_malicious_detected}")
        return pd.DataFrame(output_rows), total_malicious_detected

    # ---------------- Online mode ----------------
    elif mode == "online":
        if "scapy" not in globals():
            raise ImportError("[ERROR] scapy is required for online mode.")

        sample_counter = 0
        total_malicious_detected = 0
        output_rows = []

        def extract_packet_features(pkt):
            features = {
                "Protocol": pkt.proto if hasattr(pkt, "proto") else 0,
                "SrcPort": pkt.sport if hasattr(pkt, "sport") else 0,
                "DstPort": pkt.dport if hasattr(pkt, "dport") else 0,
                "PacketLen": len(pkt),
                "TTL": pkt.ttl if hasattr(pkt, "ttl") else 0,
                "Flags": pkt.flags.value if hasattr(pkt, "flags") else 0
            }
            return features

        def process_packet(pkt):
            nonlocal sample_counter, total_malicious_detected, output_rows
            sample_counter += 1

            features_dict = extract_packet_features(pkt)
            for f in feature_list:
                if f not in features_dict:
                    features_dict[f] = 0.0

            # ---------- ARF prediction ----------
            arf_pred = arf.predict_one(features_dict)
            arf_probs = arf.predict_proba_one(features_dict) or {}
            arf_prob_class1 = float(arf_probs.get(1, 0))

            # Decode predicted label
            def decode_label(val):
                try:
                    return encoder.inverse_transform([int(val)])[0]
                except Exception:
                    return "Unknown"

            arf_label_str = decode_label(arf_pred) if arf_pred is not None else "Unknown"

            output_rows.append({
                "arf_pred": arf_label_str,
                "arf_prob_1": arf_prob_class1,
            })

            # Log/store malicious detections
            if arf_label_str.lower() not in ["benign", "normal"]:
                total_malicious_detected += 1
                row_dict = {c: v for c, v in features_dict.items()}
                row_dict["Label"] = arf_label_str
                append_new_samples_to_master(pd.DataFrame([row_dict]))

                # Incremental online learning
                arf.learn_one(features_dict, arf_label_str)
                save_model(arf, ARF_PATH, use_cloudpickle=True)

        print("[INFO] Starting online real-time packet capture (ARF only)...")
        sniff(prn=process_packet, store=False)

        if total_malicious_detected == 0 and user_email:
            send_email(
                "[INFO] Online scan complete — No malicious traffic found",
                f"Processed {sample_counter} packets. All traffic normal.",
                to_addr=user_email
            )

        pd.DataFrame(output_rows).to_csv("online_predictions_arf.csv", index=False)
        print("[INFO] Online predictions saved to online_predictions_arf.csv")

    else:
        raise ValueError(f"[ERROR] Unknown mode: {mode}")

# ---------------- Web/UI wrapper ----------------
def run_prediction_web(input_file_path: str = None,
                       output_file_path: str = "predictions_web.csv",
                       mode: str = "offline",
                       user_email: str = None):
    if mode == "offline":
        process_stream(mode="offline",
                       input_path=input_file_path,
                       output_path=output_file_path,
                       per_detection_email=True,
                       user_email=user_email)
    elif mode == "online":
        raise ValueError("[ERROR] Online mode is not supported via web UI. Run in terminal instead.")
    else:
        raise ValueError(f"[ERROR] Unknown mode: {mode}")

# ---------------- Optional CLI ----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IDS Offline/Online Prediction")
    parser.add_argument("--mode", type=str, default="offline", choices=["offline", "online"])
    parser.add_argument("--input", type=str, help="Input CSV for offline mode")
    parser.add_argument("--email", type=str, help="Email to send alerts/results")
    args = parser.parse_args()

    if not args.input:
        print("[INFO] Running in UI-integrated mode — waiting for UI to pass input.")
    else:
        run_prediction_web(
            input_file_path=args.input,
            mode=args.mode,
            user_email=args.email
        )
