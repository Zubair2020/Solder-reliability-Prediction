import streamlit as st
import torch
import numpy as np
from model_utils import LifetimePredictionModel, parse_size, parse_ratio, normalize_feature, denormalize_feature
from sklearn.preprocessing import LabelEncoder
import time
import plotly.graph_objects as go
import base64

# Function to load the SVG file and encode it in base64
def get_svg_image(path):
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

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
        width: calc(100% - 220px);
        display: inline-block;
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
    .card {{
        background-color: #f9f9f9;
        padding: 20px;
        border: 1px solid #ddd;
        border-radius: 12px;
        box-shadow: 2px 2px 15px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
        font-family: 'Arial', sans-serif;
    }}
    .top-right {{
        position: absolute;
        top: -40px;
        right: 10px;
        width: 160px;
        height: 150px;
        z-index: 1000;
    }}
    </style>

""", unsafe_allow_html=True)

# Load the trained PyTorch model
MODEL_PATH = "saved_models/final_model.pth"

# Load model parameters
NUM_SOLDER_TYPES = 4
NUM_LED_NAMES = 7
NUM_SUBMOUNTS = 2
NUM_PAD_COUNTS = 2
NUM_NUMERICAL_FEATURES = 7
EMBEDDING_DIM = 8

# Load the trained model
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

# Load encoders for categorical features
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
    "SAC396+Sb": 22.31338,
    "SAC107+BiIn": -23.20746,
    "SAC387+SbBiNi": -20.30861
}

# Add tabs
tab1, tab2, tab3 = st.tabs(["THI Prediction Model", "MTS Prediction Model", "Xitaso Prediction Model"])

# Tab 1: LED Lifetime Prediction
with tab1:
    st.markdown('<h1 class="main-title">LED Lifetime Prediction</h1>', unsafe_allow_html=True)

    # Instructions
    with st.expander("**Instructions**", expanded=True):
        st.write("""
        Welcome to the LED Lifetime Prediction Tool! Follow the steps below to get started:
        1. Select the **Solder Type** and **LED Name** from the dropdown.
        2. Specify the **Submount Type** and **Pad Count** for your LED assembly.
        3. Enter values for **Creep Strain**, **Pad Size**, and **Pad Gap** using the sliders.
        4. Optionally, provide advanced parameters like **Ceramic Size**, **Electrical Pad Size**, **Thermal Pad Size**, and **Pad Ratio**.
        5. Hit **Predict Lifetime** to estimate how long your LED assembly will last.
        """)

    # Input fields for categorical variables
    st.markdown('<div class="section-title">Select Categorical Values</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        solder_type = st.selectbox('Solder Type', encoder_solder.classes_, help="Select the type of solder used.")
    with col2:
        led_name = st.selectbox('LED Name', encoder_led.classes_, help="Select the LED model name.")

    col3, col4 = st.columns(2)
    with col3:
        submount = st.selectbox('Submount Type', encoder_submount.classes_, help="Select the type of submount.")
    with col4:
        pad_count = st.selectbox('Pad Count', encoder_pad_count.classes_, help="Number of pads in the assembly.")

    # Input fields for numerical variables with sliders
    st.markdown('<div class="section-title">Input Numerical Values</div>', unsafe_allow_html=True)

    creep_strain = st.slider('Creep Strain', min_value=0.0, max_value=1.0, value=0.01, help="Set the creep strain (0.0 to 1.0).")
    pad_size = st.slider('Pad Size (mm²)', min_value=0.0, max_value=5.0, value=1.8, help="Set the size of the pad in mm².")
    pad_gap = st.slider('Pad Gap (mm)', min_value=0.0, max_value=1.0, value=0.3, help="Set the gap between pads in mm.")

    # Advanced options
    with st.expander("Advanced Input Settings (Optional)", expanded=False):
        ceramic_size_str = st.text_input('Ceramic Size (mm²) (e.g., 1.6 x 1.6)', value='1.6 x 1.6')
        elec_pad_size_str = st.text_input('Electrical Pad Size (mm²) (e.g., 1.5 x 0.6)', value='1.5 x 0.6')
        therm_pad_size_str = st.text_input('Thermal Pad Size (mm²) (e.g., 0)', value='0')
        pad_ratio_str = st.text_input('Pad Ratio (e.g., 1:1)', value='1:1')

    # Predict button
    if st.button("Predict Lifetime"):
        with st.spinner('Calculating LED Lifetime...'):
            time.sleep(1)

            # Convert advanced inputs to numerical
            ceramic_size = parse_size(ceramic_size_str)
            elec_pad_size = parse_size(elec_pad_size_str)
            therm_pad_size = parse_size(therm_pad_size_str)
            pad_ratio = parse_ratio(pad_ratio_str)

            # Normalize numerical features
            creep_strain_normalized = normalize_feature(creep_strain, *min_max_values['Creep_Strain'])
            pad_size_normalized = normalize_feature(pad_size, *min_max_values['Pad_Size'])
            pad_gap_normalized = normalize_feature(pad_gap, *min_max_values['Pad_gap'])
            ceramic_size_normalized = normalize_feature(ceramic_size, *min_max_values['ceramic_size'])
            elec_pad_size_normalized = normalize_feature(elec_pad_size, *min_max_values['elec_pad_size'])
            therm_pad_size_normalized = normalize_feature(therm_pad_size, *min_max_values['therm_pad_size'])
            pad_ratio_normalized = normalize_feature(pad_ratio, *min_max_values['Pad_ratio'])

            # Prepare input data for prediction
            solder_input = torch.tensor([encoder_solder.transform([solder_type])], dtype=torch.long)
            led_input = torch.tensor([encoder_led.transform([led_name])], dtype=torch.long)
            submount_input = torch.tensor([encoder_submount.transform([submount])], dtype=torch.long)
            pad_count_input = torch.tensor([encoder_pad_count.transform([pad_count])], dtype=torch.long)

            numerical_input = torch.tensor(
                [[creep_strain_normalized, pad_size_normalized, pad_gap_normalized,
                  ceramic_size_normalized, elec_pad_size_normalized, therm_pad_size_normalized, pad_ratio_normalized]],
                dtype=torch.float32
            )

            # Make prediction
            with torch.no_grad():
                prediction_normalized = model(
                    solder_input,
                    led_input,
                    submount_input,
                    pad_count_input,
                    numerical_input
                ).item()

            # Denormalize the predicted lifetime
            predicted_lifetime = denormalize_feature(prediction_normalized, *min_max_values['Lifetime'])

        # Display predicted lifetime
        st.markdown(f'<div class="prediction-result">Predicted LED Lifetime: {predicted_lifetime:.2f} cycles</div>', unsafe_allow_html=True)

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_lifetime,
            title={'text': "Predicted LED Lifetime (Cycles)"},
            gauge={
                'axis': {'range': [None, max(min_max_values['Lifetime'])]},
                'bar': {'color': "#4CAF50"},
                'steps': [
                    {'range': [0, 600], 'color': "#FF4C4C"},
                    {'range': [600, 1000], 'color': "#FFC107"},
                    {'range': [1000, 1817], 'color': "#4CAF50"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': predicted_lifetime
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: MTS Prediction Model
with tab2:
    st.markdown('<h1 class="main-title">MTS Model Prediction</h1>', unsafe_allow_html=True)

    selected_component = st.selectbox("Select Component Type", list(component_coefficients.keys()))
    selected_solder = st.selectbox("Select Solder Type", list(solder_coefficients.keys()))

    # Calculate cycle numbers for all components and solder types
    all_cycle_numbers = {}
    for component in component_coefficients:
        all_cycle_numbers[component] = {}
        for solder in solder_coefficients:
            component_factor = component_coefficients[component]
            solder_factor = solder_coefficients[solder]
            cycle_number = mts_base_cycle + component_factor + solder_factor
            all_cycle_numbers[component][solder] = cycle_number

    # Generate the graph
    fig = go.Figure()

    # Add all components and solder types to the graph
    for component in all_cycle_numbers:
        for solder in all_cycle_numbers[component]:
            cycle_number = all_cycle_numbers[component][solder]
            fig.add_trace(go.Bar(
                x=[f"{component} - {solder}"],
                y=[cycle_number],
                name=f"{component} - {solder}",
                hoverinfo='x+y',
                marker=dict(color='lightblue')  # Default color
            ))

    # Highlight the selected component and solder type
    highlighted_cycle_number = all_cycle_numbers[selected_component][selected_solder]
    fig.add_trace(go.Bar(
        x=[f"{selected_component} - {selected_solder}"],
        y=[highlighted_cycle_number],
        name=f"Selected: {selected_component} - {selected_solder}",
        marker=dict(color='red'),  # Highlight color
        hoverinfo='x+y'
    ))

    # Update graph layout for better display
    fig.update_layout(
        title={
            'text': "Cycle Numbers by Component and Solder Type",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24}
        },
        xaxis_title="Component - Solder Type",
        yaxis_title="Cycle Number",
        xaxis=dict(
            tickangle=-45,
            title_font=dict(size=18),
            tickfont=dict(size=12),
            automargin=True
        ),
        yaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=12)
        ),
        height=800,  # Larger height for better visibility
        margin=dict(l=50, r=50, t=100, b=200),  # Improved spacing
        showlegend=False,  # Hides legend for cleaner display
        plot_bgcolor='rgba(240,240,240,1)',  # Light gray background for better contrast
        hoverlabel=dict(font_size=14)  # Better hover label visibility
    )

    # Display the graph
    st.plotly_chart(fig, use_container_width=True)

    # Display highlighted prediction
    if st.button("Predict Lifetime (MTS)"):
        st.markdown(f"### **Predicted Cycle Number: {highlighted_cycle_number:.2f} cycles**")


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
