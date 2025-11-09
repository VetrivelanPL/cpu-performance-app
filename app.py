import streamlit as st
import joblib
import numpy as np

# Load your pre-trained models and imputer
regressor = joblib.load('final_xgb_regressor.joblib')
classifier = joblib.load('final_xgb_classifier.joblib')
imputer = joblib.load('imputer.joblib')

st.title('CPU Performance Prediction')

st.write("Fill in CPU parameters below and receive predictions:")

cores = st.number_input("Cores", min_value=1, max_value=128, value=4)
threads = st.number_input("Threads", min_value=1, max_value=256, value=8)
maxTurboClock = st.number_input("Max Turbo Clock (GHz)", min_value=0.1, max_value=10.0, value=4.0)
baseClock = st.number_input("Base Clock (GHz)", min_value=0.1, max_value=10.0, value=3.5)
TDP = st.number_input("TDP (W)", min_value=1, max_value=500, value=65)
L1Cache = st.number_input("L1 Cache (KB)", min_value=0, max_value=16384, value=256)
L2Cache = st.number_input("L2 Cache (MB)", min_value=0.0, max_value=128.0, value=2.0)
L3Cache = st.number_input("L3 Cache (MB)", min_value=0.0, max_value=256.0, value=8.0)
manufacturer = st.selectbox("Manufacturer", ["Intel", "AMD"])
manufacturer_num = 0 if manufacturer == "Intel" else 1

totalCache = L1Cache + L2Cache + L3Cache
coreClock = cores * baseClock

# Arrange features according to training order (this must match what the model expects!)
features = np.array([[cores, threads, maxTurboClock, baseClock, TDP, L1Cache, L2Cache, L3Cache,
                      manufacturer_num, totalCache, coreClock]])
features_imputed = imputer.transform(features)

if st.button('Predict Performance'):
    perf_score = regressor.predict(features_imputed)[0]
    perf_tier = classifier.predict(features_imputed)[0]
    tier_map = {0: "Low", 1: "Medium", 2: "High"}
    st.success(f"Predicted Performance Score: {perf_score:.2f}")
    st.info(f"Predicted Performance Tier: {tier_map.get(perf_tier, 'Unknown')}")
    st.write("Classifier output probabilities:", classifier.predict_proba(features_imputed))





