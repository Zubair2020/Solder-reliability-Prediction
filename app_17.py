import streamlit as st
import torch
import numpy as np
from model_utils import LifetimePredictionModel, parse_size, parse_ratio, normalize_feature, denormalize_feature
from sklearn.preprocessing import LabelEncoder
import time
import plotly.graph_objects as go

# CSS for responsive design and global styling
st.markdown(f"""
    <style>
    html {{
        zoom: 100%;
    }}
    body {{
        max-width: 100%;
        margin: 0;
        padding: 0;
    }}
    .main-title {{
        font-size: 2.2em;
        color: #4A90E2;
        font-weight: bold;
        text-align: left;
        margin: 0;
        padding: 0;
        font-family: 'Arial', sans-serif;
    }}
    .section-title {{
        font-size: 1.6em;
        color: #333;
        font-weight: bold;
        margin-top: 30px;
        font-family: 'Arial', sans-serif;
    }}
    .stButton>button {{
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1em;
        margin-top: 20px;
        transition: background-color 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: #45a049;
    }}
    </style>
""", unsafe_allow_html=True)

# THI Prediction Model setup
MODEL_PATH = "saved_models/final_model.pth"

NUM_SOLDER_TYPES = 4
NUM_LED_NAMES = 7
NUM_SUBMOUNTS = 2
NUM_PAD_COUNTS = 2
NUM_NUMERICAL_FEATURES = 7
EMBEDDING_DIM = 8

model = LifetimePredictionModel(
    num_solder_types=NUM_SOLDER_TYPES,
    num_led_names=NUM_LED_NAMES,
    num_submounts=NUM_SUBMOUNTS,
    num_pad_counts=NUM_PAD_COUNTS,
    numerical_input_size=NUM_NUMERICAL_FEATURES,
    embedding_dim=EMBEDDING_DIM
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

encoder_solder = LabelEncoder()
encoder_led = LabelEncoder()
encoder_submount = LabelEncoder()
encoder_pad_count = LabelEncoder()

encoder_solder.classes_ = np.array(['SAC305', 'SAC105', 'SAC387+SbBiNi', 'SAC107+BiIn'])
encoder_led.classes_ = np.array(['FC-SP2', 'FC-SP3', 'FC-GB1', 'FC-GB2', 'FC-GB3', 'VTF2', 'FC-SP4'])
encoder_submount.classes_ = np.array(['thick film AlN ceramic', 'thin film AlN ceramic'])
encoder_pad_count.classes_ = np.array(['two', 'three'])

min_max_values = {
    'Creep_Strain': (0.007, 0.576),
    'Pad_Size': (1.45, 2.79),
    'Pad_gap': (0.25, 0.45),
    'ceramic_size': (1.6, 2.5),
    'elec_pad_size': (0.49, 1.6),
    'therm_pad_size': (0.0, 1.7),
    'Pad_ratio': (1.0, 4.44),
    'Lifetime': (383, 1817)
}

# MTS Model coefficients
mts_base_cycle = 531.9855
component_coefficients = {
    "BGA": 140.3583,
    "R0402": 144.0145,
    "R0805": 144.0145,
    "WLP": -428.3873
}
solder_coefficients = {
    "SAC105": 8.74496,
    "SAC387": 12.45773,
    "SA396P": 22.31338,
    "SAC107": -23.20746,
    "SA387P": -20.30861
}

# Add tabs
tab1, tab2, tab3 = st.tabs(["THI Prediction Model", "MTS Prediction Model", "Xitaso Prediction Model"])

# Tab 1: THI Prediction Model
with tab1:
    st.markdown('<h1 class="main-title">THI Lifetime Prediction Tool</h1>', unsafe_allow_html=True)

    # Enhanced Instructions
    with st.expander("**Instructions**", expanded=True):
        st.write("""
        ### Welcome to the THI Lifetime Prediction Tool!
        Use this tool to estimate the lifetime of LED assemblies based on your selected inputs. Follow the steps below:
        
        #### Step 1: Select Categorical Values
        - Choose the **Solder Type** from the dropdown menu (e.g., SAC305, SAC105).
        - Select the **LED Name** corresponding to your design.
        - Specify the **Submount Type** and the **Pad Count** for your assembly.

        #### Step 2: Provide Numerical Inputs
        - Use the sliders to set values for:
          - **Creep Strain**: Defines the deformation caused by stress over time.
          - **Pad Size (mm²)**: Specify the size of the solder pads.
          - **Pad Gap (mm)**: Distance between the solder pads.

        #### Step 3: Optional Advanced Parameters
        - Provide additional parameters like:
          - **Ceramic Size** (e.g., 1.6 x 1.6 mm).
          - **Electrical Pad Size** (e.g., 1.5 x 0.6 mm).
          - **Thermal Pad Size** and **Pad Ratio**.

        #### Step 4: Predict
        - Click the **Predict Lifetime** button to calculate the estimated lifetime based on the provided inputs.
        """)

    # Input fields for THI Prediction Model
    solder_type = st.selectbox('Solder Type', encoder_solder.classes_)
    led_name = st.selectbox('LED Name', encoder_led.classes_)
    submount = st.selectbox('Submount Type', encoder_submount.classes_)
    pad_count = st.selectbox('Pad Count', encoder_pad_count.classes_)
    creep_strain = st.slider('Creep Strain', 0.0, 1.0, 0.01)
    pad_size = st.slider('Pad Size (mm²)', 1.45, 2.79, 1.8)
    pad_gap = st.slider('Pad Gap (mm)', 0.25, 0.45, 0.3)

    if st.button("Predict Lifetime"):
        with st.spinner("Calculating LED Lifetime..."):
            numerical_input = torch.tensor([[normalize_feature(creep_strain, *min_max_values['Creep_Strain']),
                                             normalize_feature(pad_size, *min_max_values['Pad_Size']),
                                             normalize_feature(pad_gap, *min_max_values['Pad_gap'])]], dtype=torch.float32)
            solder_input = torch.tensor([encoder_solder.transform([solder_type])], dtype=torch.long)
            led_input = torch.tensor([encoder_led.transform([led_name])], dtype=torch.long)
            submount_input = torch.tensor([encoder_submount.transform([submount])], dtype=torch.long)
            pad_count_input = torch.tensor([encoder_pad_count.transform([pad_count])], dtype=torch.long)

            with torch.no_grad():
                prediction_normalized = model(solder_input, led_input, submount_input, pad_count_input, numerical_input).item()
                predicted_lifetime = denormalize_feature(prediction_normalized, *min_max_values['Lifetime'])

        st.markdown(f"### **Predicted LED Lifetime: {predicted_lifetime:.2f} cycles**")

# Tab 2: MTS Prediction Model
with tab2:
    st.markdown('<h1 class="main-title">MTS Model Prediction</h1>', unsafe_allow_html=True)

    selected_component = st.selectbox("Select Component Type", list(component_coefficients.keys()))
    selected_solder = st.selectbox("Select Solder Type", list(solder_coefficients.keys()))

    if st.button("Predict Lifetime (MTS)"):
        component_factor = component_coefficients[selected_component]
        solder_factor = solder_coefficients[selected_solder]
        cycle_number = mts_base_cycle + component_factor + solder_factor

        st.markdown(f"### **Predicted Cycle Number (MTS Model): {cycle_number:.2f} cycles**")

# Tab 3: Xitaso Prediction Model
with tab3:
    st.markdown('<h1 class="main-title">Xitaso Model</h1>', unsafe_allow_html=True)

    subtab1, subtab2 = st.tabs(["Sub-Tab 1", "Sub-Tab 2"])

    with subtab1:
        st.markdown('<h2 class="section-title">Sub-Tab 1 Content</h2>', unsafe_allow_html=True)
        st.write("This is the content for Sub-Tab 1 inside the Xitaso Model tab.")

    with subtab2:
        st.markdown('<h2 class="section-title">Sub-Tab 2 Content</h2>', unsafe_allow_html=True)
        st.write("This is the content for Sub-Tab 2 inside the Xitaso Model tab.")
