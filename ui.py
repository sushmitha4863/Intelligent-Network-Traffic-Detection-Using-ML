#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
import threading
import time
import os
from prediction import process_stream  # For offline mode

try:
    from scapy.all import sniff
except ImportError:
    st.error("⚠️ scapy is not installed. Live detection will not work.")
    sniff = None

# ----------------------- Streamlit UI -----------------------
st.set_page_config(page_title="Adaptive IDS Web UI", layout="wide")
st.title("INTELLIGENT NETWORK TRAFFFIC DETECTION USING ML")

mode = st.radio("Select Mode", ["Offline Detection", "Online Live Detection"])

feedback_file = "feedback_data.csv"

# ---------------- Visualization ----------------
def plot_prediction_counts(df):
    import plotly.express as px
    counts = df['lgbm_pred'].value_counts().reset_index()
    counts.columns = ['Attack Type', 'Count']
    fig = px.bar(counts, x='Attack Type', y='Count', color='Attack Type', title="Prediction Counts")
    st.plotly_chart(fig, use_container_width=True)

def plot_prediction_pie(df):
    import plotly.express as px
    counts = df['lgbm_pred'].value_counts().reset_index()
    counts.columns = ['Attack Type', 'Count']
    fig = px.pie(counts, values='Count', names='Attack Type', title="Prediction Distribution")
    st.plotly_chart(fig, use_container_width=True)

def save_feedback(df):
    st.subheader("Feedback (Correct any mispredicted flows)")
    df_feedback = df.copy()
    df_feedback['Correct_Label'] = df_feedback['lgbm_pred']
    edited_df = st.data_editor(df_feedback, num_rows="dynamic")
    if st.button("Submit Feedback"):
        edited_df.to_csv(
            feedback_file, mode='a', index=False,
            header=not os.path.exists(feedback_file)
        )
        st.success(f"✅ Feedback saved for {len(edited_df)} samples!")

# ---------------- Offline Mode ----------------
if mode == "Offline Detection":
    uploaded_file = st.file_uploader("Upload CSV for offline IDS detection", type="csv")
    user_email = st.text_input("Your Email (optional, for alerts)", value="")

    if uploaded_file and st.button("Run Offline Detection"):
        st.info("Running offline IDS prediction...")
        tmp_file_path = os.path.join("tmp_upload.csv")
        with open(tmp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        process_stream(
            mode="offline",
            input_path=tmp_file_path,
            output_path="predictions_web.csv",
            user_email=user_email
        )

        if os.path.exists("predictions_web.csv"):
            result_df = pd.read_csv("predictions_web.csv")
            st.success("Offline detection finished!")
            st.dataframe(result_df)
            plot_prediction_counts(result_df)
            plot_prediction_pie(result_df)
            save_feedback(result_df)

# ---------------- Online Live Mode ----------------
else:
    if sniff is None:
        st.warning("⚠️ scapy is required for live detection.")
    else:
        st.info("Live IDS detection using scapy")
        interface = st.text_input("Network interface (e.g., en0 for Mac, eth0 for Linux)", value="en0")
        packet_count = st.number_input("Packets per batch", min_value=1, max_value=50, value=10)
        batch_count = st.number_input("Number of batches", min_value=1, max_value=20, value=3)
        user_email = st.text_input("Your Email (optional, for alerts)", value="")

        start_button = st.button("Start Online Detection")

        if start_button:
            live_result = pd.DataFrame()
            progress_text = st.empty()
            table_placeholder = st.empty()
            chart_counts_placeholder = st.empty()
            chart_pie_placeholder = st.empty()

            sample_counter = 0
            for b in range(batch_count):
                progress_text.info(f"Capturing batch {b+1}/{batch_count} ...")
        
                # Capture packets
                packets = sniff(count=packet_count, iface=interface, timeout=10)
                batch_data = []
                for pkt in packets:
                    sample_counter += 1
                    features_dict = {
                        "Protocol": getattr(pkt, "proto", 0),
                        "SrcPort": getattr(pkt, "sport", 0),
                        "DstPort": getattr(pkt, "dport", 0),
                        "PacketLen": len(pkt),
                        "TTL": getattr(pkt, "ttl", 0),
                        "Flags": getattr(pkt, "flags", 0)
                    }
                    batch_data.append(features_dict)

                     # Prepare DataFrame for prediction
                batch_df = pd.DataFrame(batch_data)
                batch_df.to_csv("online_batch.csv", index=False)

                    # Run prediction (offline mode on batch CSV)
                process_stream(
                    mode="offline", 
                    input_path="online_batch.csv", 
                    output_path="online_batch_pred.csv", 
                    user_email=user_email )
                    
                    
                

                pred_df = pd.read_csv("online_batch_pred.csv")
                live_result = pd.concat([live_result, pred_df])

                    # Update UI safely in main thread
                table_placeholder.dataframe(live_result)
                plot_prediction_counts(live_result)
                plot_prediction_pie(live_result)
                time.sleep(1)

                    

            progress_text.success("✅ Online detection finished!")

   


