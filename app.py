import streamlit as st
import pandas as pd
import pickle
from pgmpy.inference import VariableElimination

def calculate_congestion_state(row):
    if row['V/C'] < 0.1:
        vc_value = 1
    elif row['V/C'] < 0.4:
        vc_value = 2
    elif row['V/C'] < 0.6:
        vc_value = 3
    elif row['V/C'] < 0.8:
        vc_value = 4
    elif row['V/C'] < 1.0:
        vc_value = 5
    else:
        vc_value = 6

    if row['SPI'] == 'very smooth':
        spi_value = 1
    elif row['SPI'] == 'smooth':
        spi_value = 2
    elif row['SPI'] == 'mild':
        spi_value = 3
    elif row['SPI'] == 'heavy':
        spi_value = 4
    else:
        spi_value = 0

    congestion_state = vc_value + spi_value
    return congestion_state

def load_model():
    try:
        with open('bayesian_model.pkl', 'rb') as f:
            model = pickle.load(f)
            return model
    except FileNotFoundError:
        st.error("Model file 'bayesian_model.pkl' not found.")
        st.stop()
    except pickle.UnpicklingError:
        st.error("Error loading model.")
        st.stop()

st.title("Traffic Congestion Predictor")
model = load_model()
inference = VariableElimination(model)

V_encoded = st.selectbox("Select V_encoded Category:", options=['Low', 'Medium', 'High'])
D_encoded = st.selectbox("Select Day of Week:", options=['Weekday', 'Weekend'])
T_encoded = st.selectbox("Select Time of Day:", options=['AM Peak', 'PM Peak', 'Off-Peak'])
speed_encoded = st.selectbox("Select Speed Category:", options=['Low', 'Medium', 'High'])

encode_mappings = {
    'V_encoded': {'Low': 1, 'Medium': 2, 'High': 3},
    'D_encoded': {'Weekday': 1, 'Weekend': 2},
    'T_encoded': {'AM Peak': 1, 'PM Peak': 2, 'Off-Peak': 3},
    'Speed_Encoded': {'Low': 1, 'Medium': 2, 'High': 3},
    'SPI_encoded': {'Very smooth': 1, 'Smooth': 2, 'Mild': 3, 'Heavy': 4},
    'V/C Level': {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6},
}

V_encoded_value = encode_mappings['V_encoded'][V_encoded]
D_encoded_value = encode_mappings['D_encoded'][D_encoded]
T_encoded_value = encode_mappings['T_encoded'][T_encoded]
Speed_Encoded_value = encode_mappings['Speed_Encoded'][speed_encoded]

speed_values = {'Low': 10, 'Medium': 25, 'High': 45}
speed = speed_values[speed_encoded]
st.session_state.speed = speed

max_speed = 57  
SPI = (speed / max_speed) * 100

spi_bins = [0, 25, 50, 75, 100]
spi_labels = ['Heavy', 'Mild', 'Smooth', 'Very smooth']
SPI_encoded_label = pd.cut([SPI], bins=spi_bins, labels=spi_labels, right=True, include_lowest=True)[0]
SPI_encoded = encode_mappings['SPI_encoded'][SPI_encoded_label]

vehicle_length = 8.84  
segment_length = 1000  
Nmax = (segment_length / vehicle_length) * 3

bus_count = {'Low': 11, 'Medium': 50, 'High': 70}
st.session_state.bus_count = bus_count[V_encoded]

VC_ratio = st.session_state.bus_count / Nmax

def classify_vc(vc):
    if vc <= 0.10:
        return 'A'
    elif vc <= 0.40:
        return 'B'
    elif vc <= 0.60:
        return 'C'
    elif vc <= 0.80:
        return 'D'
    elif vc <= 1.00:
        return 'E'
    else:
        return 'F'

VC_level_label = classify_vc(VC_ratio)
VC_level = encode_mappings['V/C Level'][VC_level_label]

st.write(f"Assigned Speed: {st.session_state.speed} km/h")
st.write(f"Calculated SPI: {SPI:.2f}")
st.write(f"SPI Category (Encoded): {SPI_encoded_label} (Encoded Value: {SPI_encoded})")
st.write(f"Simulated Bus Count: {st.session_state.bus_count}")
st.write(f"Calculated V/C Ratio: {VC_ratio:.2f}")
st.write(f"V/C Level (Encoded): {VC_level_label} (Encoded Value: {VC_level})")

st.write(f"Encoded Values -> V: {V_encoded_value}, D: {D_encoded_value}, T: {T_encoded_value}, Speed: {Speed_Encoded_value}, SPI: {SPI_encoded}, V/C Level: {VC_level}")

if st.button("Predict Congestion State"):
    query = {
        'V_encoded': V_encoded_value,
        'D_encoded': D_encoded_value,
        'T_encoded': T_encoded_value,
        'Speed_Encoded': Speed_Encoded_value,
        'SPI_encoded': SPI_encoded,
        'V/C Level': VC_level
    }

    try:
        result = inference.map_query(variables=['Congestion_State'], evidence=query)
        predicted_congestion_state = SPI_encoded + VC_level
        
        if predicted_congestion_state < 4:
            congestion_level = "Low"
        elif predicted_congestion_state in [4, 5]:
            congestion_level = "Mild"
        else:
            congestion_level = "Heavy"
        
        st.success(f"Predicted Congestion State: {predicted_congestion_state}")
        st.info(f"Congestion Level: {congestion_level}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
