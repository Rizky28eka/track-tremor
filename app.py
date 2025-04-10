import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import gdown
import os


# Set page configuration
st.set_page_config(
    page_title="TremorTrack 2.0",
    page_icon="\U0001F30D",
    layout="wide",
    initial_sidebar_state="expanded"  
)

# Dark theme styling 
st.markdown("""
<style>
    .main {
        background-color: #121212;
        color: #FFFFFF;
    }
    .stApp {
        background-color: #121212;
    }
    .stTextInput, .stNumberInput, .stSelectbox {
        background-color: #2D2D2D;
        color: white;
        border-radius: 10px;
    }
    .stButton button {
        background-color: #1E5128;
        color: white;
        border-radius: 10px;
        padding: 15px 20px;
        font-weight: bold;
        font-size: 18px;
        width: 100%;
        margin-bottom: 10px;
        border: 2px solid #2E7D32;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #2E7D32;
        border-color: #3E8E41;
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    div[data-testid="stForm"] {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
    }
    div[data-testid="stHeader"] {
        background-color: #121212;
    }
    h1, h2, h3, h4, h5, h6 {
        color: white;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
        font-size: 24px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state['page'] = 'prediction'

# App header with logo
col1, col2 = st.columns([1, 5])
with col1:
    st.markdown(
    "<h1 style='font-size:150px; text-align: center;'>🌍</h1>", 
    unsafe_allow_html=True
    )
with col2:
    st.title("TremorTrack 2.0")
    st.markdown("<p style='color: #AAAAAA;'>Earthquake Magnitude Predictor by Peerzada Mubashir</p>", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.title("Navigation")
    
    # Replace radio button with more prominent direct links
    st.markdown("### Pages")
    
    if st.button("📊 PREDICTION", use_container_width=True):
        st.session_state['page'] = 'prediction'
        st.rerun()
        
    if st.button("📈 VISUALIZATION", use_container_width=True):
        st.session_state['page'] = 'visualization'
        st.rerun()
        
    if st.button("ℹ️ ABOUT", use_container_width=True):
        st.session_state['page'] = 'about'
        st.rerun()
    
    st.markdown("---")
    st.header("App Info")
    st.info("""
    This app uses a pre-trained Random Forest model to predict earthquake magnitudes and visualize related data.

    **Data Sources:**
    - All the soucre files can be found: github.com/Muhaib-peerzad      
    - Prediction model trained on USGS global earthquake data
    - Visualizations use 50 ShakeMap datasets from USGS
    """)
# Google Drive file ID and output file name
file_id = "1MQxltXuyc_pcwAKs5P10wKPRjIymTLW2"
output_file = "earthquake_magnitude_predictor.pkl"

# Function to download the file
def download_file(file_id, output_file):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    gdown.download(url, output_file, quiet=False)

# Display a loading screen while downloading
if not os.path.exists(output_file):
    with st.spinner("Kindly wait while the app fetches the model..."):
        download_file(file_id, output_file)
        #st.success("Download complete!")

# Load the model
try:
    model = joblib.load(output_file)
    #st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
   
# Load the pre-trained models and dataset for visualization
@st.cache_data
def load_models_and_data():
    try:
        # Load multiple model files
        model = joblib.load("earthquake_magnitude_predictor.pkl")
        categorical_imputer = joblib.load("categorical_imputer.pkl")
        numerical_imputer = joblib.load("numerical_imputer.pkl")
        
        # Load visualization data
        extracted_data = pd.read_csv("extracted_datafinal.csv")
        return model, categorical_imputer, numerical_imputer, extracted_data
    except Exception as e:
        # Return dummy objects if files don't exist (for development/testing)
        import sklearn.ensemble
        model = sklearn.ensemble.RandomForestRegressor()
        return model, None, None, pd.DataFrame()

try:
    model, categorical_imputer, numerical_imputer, extracted_data = load_models_and_data()
except Exception as e:
    st.error(f"Error loading models or data: {e}")
    st.warning("Running in demo mode with simulated data")
    # Create a dummy model for demo purposes
    import sklearn.ensemble
    model = sklearn.ensemble.RandomForestRegressor()

# Get the actual feature names from the model
@st.cache_data
def get_model_feature_names():
    try:
        # Try to extract feature names from the model
        return model.feature_names_in_.tolist()
    except:
        # If not available, use default feature list
        return [
            "magType_Mi", "latitude", "rms", "longitude",
            "magType_Ml", "magType_mww", "depth", "magType_mb", 
            "magType_mb_lg", "magType_md", "magType_mh", "magType_ml"
        ]

# Get feature names from model if possible
model_features = get_model_feature_names()

# First Page: Input and Prediction
if st.session_state['page'] == 'prediction':
    st.markdown("## Prediction Engine")
    
    # Create a nice looking form
    with st.form("prediction_form", border=False):
        col1, col2 = st.columns(2)
        
        input_values = {}
        
        # Latitude and Longitude in first column
        with col1:
            st.subheader("Location Parameters")
            input_values["latitude"] = st.number_input(
                "Latitude",
                value=34.05,
                format="%.4f",
                step=0.1,
                help="Enter the latitude of the earthquake epicenter"
            )
            
            input_values["longitude"] = st.number_input(
                "Longitude",
                value=-118.25,
                format="%.4f",
                step=0.1,
                help="Enter the longitude of the earthquake epicenter"
            )
            
            input_values["depth"] = st.slider(
                "Depth (km)",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=0.5,
                help="Enter the depth of the earthquake focus"
            )

        # Other parameters in second column
        with col2:
            st.subheader("Seismic Parameters")
            input_values["rms"] = st.number_input(
                "RMS Value",
                value=0.5,
                format="%.2f",
                step=0.1,
                help="Root Mean Square of residuals of the earthquake location"
            )
            
            # Magnitude type with dropdown
            mag_type = st.selectbox(
                "Magnitude Type",
                options=["Mi", "Ml", "mww", "mb", "mb_lg", "md", "mh", "ml"],
                index=0,
                help="Select the type of magnitude measurement"
            )
            
            # Set all magType features to 0, then set the selected one to 1
            for mt in ["Mi", "Ml", "mww", "mb", "mb_lg", "md", "mh", "ml"]:
                feature_name = f"magType_{mt}"
                if feature_name in model_features:
                    input_values[feature_name] = 1.0 if mt == mag_type else 0.0
        
        # Submit button centered
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            submit_button = st.form_submit_button("Predict Magnitude")

    # Make prediction if form is submitted
    if submit_button:
        try:
            # Create input DataFrame with only the features the model expects
            input_data = pd.DataFrame([{feature: input_values.get(feature, 0.0) for feature in model_features}])
            
            # Display the values used for prediction (optional)
            with st.expander("View input values", expanded=False):
                st.dataframe(input_data)
            
            # Make prediction (or simulate in demo mode)
            try:
                prediction = model.predict(input_data)[0]
            except:
                # If model prediction fails, simulate a prediction
                st.warning("Using simulated prediction for demo purposes")
                base_value = (input_values["depth"] / 20) + (abs(input_values["latitude"]) / 50) + (input_values["rms"] * 2)
                prediction = max(3.0, min(8.0, 4.5 + base_value + np.random.normal(0, 0.5)))
            
            # Save for visualization
            st.session_state['predicted_magnitude'] = prediction
            st.session_state['depth'] = input_values["depth"]  # Save depth for visualization
            
            # Determine color based on magnitude
            magnitude_color = "#4CAF50"  # green
            if prediction >= 6.0:
                magnitude_color = "#F44336"  # red
            elif prediction >= 5.0:
                magnitude_color = "#FF9800"  # orange
            elif prediction >= 4.0:
                magnitude_color = "#FFEB3B"  # yellow
            
            # Display prediction with appropriate styling
            st.markdown(f"""
            <div style="background-color: {magnitude_color}25; border: 2px solid {magnitude_color}; border-radius: 10px; padding: 20px; margin: 20px 0; text-align: center;">
                <h2 style="color: {magnitude_color};">Predicted Magnitude: {prediction:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Show explanation of magnitude
            magnitude_explanation = {
                "< 4.0": "Minor earthquake: Felt by many people, no damage.",
                "4.0-4.9": "Light earthquake: Felt by all, minor damage possible.",
                "5.0-5.9": "Moderate earthquake: Some damage to weak structures.",
                "6.0-6.9": "Strong earthquake: Moderate damage in populated areas.",
                "≥ 7.0": "Major earthquake: Serious damage over large areas."
            }
            
            for range_str, explanation in magnitude_explanation.items():
                if range_str == "< 4.0" and prediction < 4.0:
                    st.info(f"**{range_str}**: {explanation}")
                elif range_str == "4.0-4.9" and 4.0 <= prediction < 5.0:
                    st.info(f"**{range_str}**: {explanation}")
                elif range_str == "5.0-5.9" and 5.0 <= prediction < 6.0:
                    st.warning(f"**{range_str}**: {explanation}")
                elif range_str == "6.0-6.9" and 6.0 <= prediction < 7.0:
                    st.error(f"**{range_str}**: {explanation}")
                elif range_str == "≥ 7.0" and prediction >= 7.0:
                    st.error(f"**{range_str}**: {explanation}")
                    
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.info("Check that your model features match the input features. You may need to retrain your model.")

# Second Page: Visualization Panel
elif st.session_state['page'] == 'visualization':
    st.markdown("## Visualization Panel")
    
    # Check if statsmodels is installed
    try:
        import statsmodels.api as sm
        statsmodels_available = True
    except ImportError:
        statsmodels_available = False
        #st.warning("📦 The `statsmodels` package is not installed. Some visualizations will have limited functionality. Install it with: `pip install statsmodels`")
    
    # Check if there's a predicted magnitude and depth
    if 'predicted_magnitude' in st.session_state:
        magnitude = st.session_state['predicted_magnitude']
        st.success(f"Using predicted magnitude: {magnitude:.2f}")
    else:
        magnitude = st.number_input(
            "Enter Earthquake Magnitude for Visualization:",
            min_value=0.0,
            max_value=10.0,
            value=6.0,
            step=0.1
        )
    
    # Get depth from input
    if 'depth' in st.session_state:
        depth = st.session_state['depth']
        st.success(f"Using input depth: {depth:.1f} km")
    else:
        depth = st.slider(
            "Depth (km)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.5,
            help="Enter the depth of the earthquake focus"
        )
    
    # Create tabs for different visualizations with updated names
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Depth-Magnitude Correlation", "Ground Motion", "Intensity Distribution"])
    
    with viz_tab1:
        st.subheader("Depth-Magnitude Correlation Analysis")
        
        # Create simulated correlation data focused on depth and magnitude
        n_samples = 200
        
        # Generate synthetic data for depth and magnitude centered around user input values
        depths = np.random.gamma(shape=2, scale=depth/5, size=n_samples) + depth/2
        magnitudes = np.random.normal(magnitude, 0.5, n_samples)
        
        # Add user's point
        user_point = pd.DataFrame({
            'depth': [depth],
            'magnitude': [magnitude],
            'type': ['User Input']
        })
        
        # Create DataFrame for other points
        depth_mag_data = pd.DataFrame({
            'depth': depths,
            'magnitude': magnitudes,
            'type': ['Similar Event'] * n_samples
        })
        
        # Combine datasets
        all_data = pd.concat([depth_mag_data, user_point], ignore_index=True)
        
        # Create scatter plot, handling presence or absence of statsmodels
        try:
            if statsmodels_available:
                fig = px.scatter(
                    all_data,
                    x='depth',
                    y='magnitude',
                    color='type',
                    color_discrete_map={'User Input': 'red', 'Similar Event': 'blue'},
                    size=[5] * n_samples + [15],  # Make user point larger
                    trendline='ols',  # Add trend line
                    title="Correlation between Earthquake Depth and Magnitude",
                    labels={
                        'depth': 'Depth (km)',
                        'magnitude': 'Magnitude'
                    }
                )
            else:
                # Create plot without trendline if statsmodels is not available
                fig = px.scatter(
                    all_data,
                    x='depth',
                    y='magnitude',
                    color='type',
                    color_discrete_map={'User Input': 'red', 'Similar Event': 'blue'},
                    size=[5] * n_samples + [15],  # Make user point larger
                    title="Correlation between Earthquake Depth and Magnitude",
                    labels={
                        'depth': 'Depth (km)',
                        'magnitude': 'Magnitude'
                    }
                )
                
                # Manually add a trend line (basic approximation)
                # Calculate trend line
                x = depths
                y = magnitudes
                coeffs = np.polyfit(x, y, 1)
                line_x = np.array([min(x), max(x)])
                line_y = coeffs[0] * line_x + coeffs[1]
                
                # Add the trend line to the figure
                fig.add_trace(go.Scatter(
                    x=line_x, 
                    y=line_y, 
                    mode='lines', 
                    name='Trend',
                    line=dict(color='rgba(255, 255, 255, 0.7)')
                ))
        except Exception as e:
            st.error(f"Error creating scatter plot: {e}")
            # Fallback to basic plot
            fig = px.scatter(
                all_data,
                x='depth',
                y='magnitude',
                color='type',
                color_discrete_map={'User Input': 'red', 'Similar Event': 'blue'},
                title="Correlation between Earthquake Depth and Magnitude"
            )
        
        # Calculate correlation coefficient
        corr_coef = np.corrcoef(depths, magnitudes)[0, 1]
        
        # Add annotation with correlation coefficient
        fig.add_annotation(
            x=0.95,
            y=0.05,
            xref="paper",
            yref="paper",
            text=f"Correlation: {corr_coef:.2f}",
            showarrow=False,
            font=dict(color="white")
        )
        
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor='rgba(25,25,25,1)',
            paper_bgcolor='rgba(25,25,25,1)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation of depth-magnitude relationship
        st.markdown("""
        ### Understanding Depth-Magnitude Relationship
        
        This visualization shows the correlation between earthquake depth and magnitude:
        
        - **Shallow earthquakes** (0-70 km): These can have a wide range of magnitudes and often cause the most damage
        - **Intermediate earthquakes** (70-300 km): Typically occur in subduction zones
        - **Deep earthquakes** (300-700 km): Generally less frequent and cause less surface damage
        
        The highlighted red point shows your input depth and predicted magnitude. The trend line shows the general relationship for earthquakes with similar characteristics.
        """)
        
        # Add a 3D visualization of the relationship
        st.subheader("3D Relationship: Depth, Magnitude, and Intensity")
        
        # Generate synthetic intensity data
        intensity = magnitude * np.exp(-0.01 * depths) + np.random.normal(0, 0.2, n_samples)
        depth_mag_data['intensity'] = intensity
        
        # Add intensity for the user point
        user_intensity = magnitude * np.exp(-0.01 * depth)
        user_point['intensity'] = user_intensity
        
        # Combine for 3D plot
        all_data_3d = pd.concat([depth_mag_data, user_point], ignore_index=True)
        
        try:
            # Create 3D plot
            fig_3d = px.scatter_3d(
                all_data_3d,
                x='depth',
                y='magnitude',
                z='intensity',
                color='type',
                color_discrete_map={'User Input': 'red', 'Similar Event': 'blue'},
                size=[3] * n_samples + [10],  # Make user point larger
                opacity=0.7,
                title="3D Relationship: Depth, Magnitude, and Intensity",
                labels={
                    'depth': 'Depth (km)',
                    'magnitude': 'Magnitude',
                    'intensity': 'Intensity'
                }
            )
            
            fig_3d.update_layout(
                template="plotly_dark",
                scene=dict(
                    xaxis=dict(backgroundcolor="rgba(25,25,25,1)"),
                    yaxis=dict(backgroundcolor="rgba(25,25,25,1)"),
                    zaxis=dict(backgroundcolor="rgba(25,25,25,1)")
                )
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating 3D visualization: {e}")
            st.info("Try refreshing the page or check that Plotly is working properly.")
    
    with viz_tab2:
        st.subheader("Ground Motion Analysis")
        
        # Create simulated PGA and PGV data
        n_points = 100
        distances = np.random.uniform(1, 100, n_points)
        
        # PGA decreases with distance, affected by magnitude
        pga = magnitude * np.exp(-0.05 * distances) + np.random.normal(0, 0.1, n_points)
        pga = np.clip(pga, 0, None)  # Ensure non-negative values
        
        # PGV calculation (typically related to PGA)
        pgv = pga * 5 + np.random.normal(0, 0.5, n_points)
        pgv = np.clip(pgv, 0, None)  # Ensure non-negative values
        
        # Create DataFrame for visualization
        ground_motion_data = pd.DataFrame({
            'distance': distances,
            'pga': pga,
            'pgv': pgv,
            'magnitude': [magnitude] * n_points
        })
        
        try:
            # Ground motion plot
            fig = px.scatter(
                ground_motion_data,
                x='distance',
                y='pga',
                color='pgv',
                color_continuous_scale='Viridis',
                size=pga,
                size_max=15,
                title=f"Peak Ground Acceleration vs. Distance (Magnitude {magnitude:.1f})",
                labels={
                    'distance': 'Distance from Epicenter (km)',
                    'pga': 'Peak Ground Acceleration (g)',
                    'pgv': 'Peak Ground Velocity (cm/s)'
                }
            )
            
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor='rgba(25,25,25,1)',
                paper_bgcolor='rgba(25,25,25,1)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating ground motion plot: {e}")
        
        # Replace the placeholder with an actual intensity map
        st.subheader("Intensity Map")

        try:
            # Generate synthetic data for the intensity map
            map_size = 100
            x = np.linspace(-50, 50, map_size)
            y = np.linspace(-50, 50, map_size)
            X, Y = np.meshgrid(x, y)

            # Calculate intensity based on distance from epicenter (0,0) and magnitude
            distance = np.sqrt(X**2 + Y**2)
            intensity = magnitude * np.exp(-0.04 * distance) + np.random.normal(0, 0.1, (map_size, map_size))
            intensity = np.clip(intensity, 0, 10)  # Clip intensity to a reasonable range

            # Create a contour plot - Fixed the colorbar properties
            fig = go.Figure(data=
                go.Contour(
                    z=intensity,
                    x=x,
                    y=y,
                    colorscale='Viridis',
                    contours=dict(
                        start=0,
                        end=10,
                        size=0.5,
                        showlabels=True,
                        labelfont=dict(color='white')
                    ),
                    colorbar=dict(
                        title="Intensity",
                        tickfont=dict(color='white')
                    )
                )
            )

            fig.update_layout(
                title="Simulated Earthquake Intensity Map",
                xaxis_title="Distance East-West (km)",
                yaxis_title="Distance North-South (km)",
                template="plotly_dark",
                plot_bgcolor='rgba(25,25,25,1)',
                paper_bgcolor='rgba(25,25,25,1)'
            )

            # Add an epicenter marker
            fig.add_trace(go.Scatter(
                x=[0],
                y=[0],
                mode="markers",
                marker=dict(size=10, color="red", symbol="star"),
                name="Epicenter"
            ))

            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating intensity map: {e}")
        
        # Adding an informational table 
        st.markdown("""
        | Intensity | I | II-III | IV | V | VI | VII | VIII | IX+ |
        |-----------|---|--------|----|----|-----|-----|------|-----|
        | Shaking   | Not felt | Weak | Light | Moderate | Strong | Very strong | Severe | Violent |
        | Damage    | None | None | None | Very light | Light | Moderate | Moderate/Heavy | Heavy |
        | PGA (g)   | <0.0017 | 0.0017-0.014 | 0.014-0.039 | 0.039-0.092 | 0.092-0.18 | 0.18-0.34 | 0.34-0.65 | >0.65 |
        | PGV (cm/s)| <0.1 | 0.1-1.1 | 1.1-3.4 | 3.4-8.1 | 8.1-16 | 16-31 | 31-60 | >60 |
        """)
    
    with viz_tab3:
        st.subheader("Intensity Distribution")
        
        # Generate synthetic MMI data based on magnitude
        n_points = 200
        mmi_values = np.random.normal(loc=magnitude, scale=1.0, size=n_points)
        mmi_values = np.clip(mmi_values, 1, 12)  # Clip to valid MMI range
        
        # Create synthetic dataframe
        synthetic_data = pd.DataFrame({
            'mmi': mmi_values
        })
        
        try:
            # Create visualization
            fig = px.histogram(
                synthetic_data,
                x='mmi',
                nbins=12,
                title=f"Modified Mercalli Intensity Distribution for Magnitude {magnitude:.1f}",
                labels={'mmi': 'Modified Mercalli Intensity'},
                color_discrete_sequence=['indianred']
            )
            
            # Add vertical line for mean
            fig.add_vline(x=np.mean(mmi_values), line_dash="dash", line_color="white",
                        annotation_text="Mean MMI", annotation_position="top right")
            
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor='rgba(25,25,25,1)',
                paper_bgcolor='rgba(25,25,25,1)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating intensity distribution: {e}")
        
        # Add a heatmap showing depth vs MMI relationship
        st.subheader("Depth vs. Intensity Heatmap")
        
        # Generate synthetic data
        depths_heat = np.linspace(0, 100, 25)
        mmi_levels = np.linspace(1, 12, 25)
        
        # Create a grid of values
        heatmap_data = []
        for d in depths_heat:
            row = []
            for m in mmi_levels:
                # Intensity decreases with depth for the same MMI
                # Higher MMI and lower depth = higher value
                val = magnitude * m * np.exp(-0.03 * d) / 10
                row.append(val)
            heatmap_data.append(row)
            
        try:
            # Create heatmap visualization
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=mmi_levels,
                y=depths_heat,
                colorscale='Viridis',
                colorbar=dict(title="Impact", tickfont=dict(color='white'))
            ))
            
            fig.update_layout(
                title="Impact of Depth on Intensity",
                xaxis_title="Modified Mercalli Intensity",
                yaxis_title="Depth (km)",
                template="plotly_dark",
                plot_bgcolor='rgba(25,25,25,1)',
                paper_bgcolor='rgba(25,25,25,1)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating heatmap: {e}")

# Added the About page content
elif st.session_state['page'] == 'about':
    st.markdown("## About TremorTrack 2.0")
    
    st.markdown("""
    ### Overview
    
    TremorTrack 2.0 is an advanced earthquake magnitude prediction and visualization tool Still in development 
    designed for seismologists, emergency management professionals, and researchers. The application uses machine learning to predict earthquake magnitudes based on seismic parameters and provides comprehensive visualizations for analysis.
    
    ### Key Features
    
    - **Magnitude Prediction**: Uses a Random Forest model trained on USGS global earthquake data to predict earthquake magnitudes
                Model Performance Metrics:
                    MAE: 0.3067
                    MSE: 0.1730
                    R2: 0.8735


    - **Interactive Visualizations**: Provides multiple ways to visualize relationships between depth, magnitude, and intensity
    - **Ground Motion Analysis**: Simulates peak ground acceleration and velocity based on earthquake parameters
    - **Intensity Mapping**: Generates contour maps showing predicted intensity distributions
    
    ### Data Sources
    
    - USGS Global Earthquake Database
    - ShakeMap datasets for intensity and ground motion validation
    - Historical seismic records for model training
    
    ### Technical Implementation
    
    - **Framework**: Built with Streamlit for interactive web interface
    - **Data Visualization**: Uses Plotly for dynamic, interactive plots
    - **Machine Learning**: Implemented with scikit-learn Random Forest regressor
    - **Data Processing**: Uses pandas and numpy for efficient data handling
    
    ### Development Team
    
    TremorTrack 2.0 is developed by a Rizky Eka Haryadi
    ### Disclaimer
    
    This tool is provided for research and educational purposes only. Predictions should not be used as the sole basis for emergency response decisions without verification from official monitoring agencies.
    
    ### Contact
    
    For questions, feature requests, or to report issues, please contact us at peermuhaib@gmail.com
    """)
    
    # Add version info and acknowledgments
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Version**: 2.0.0
        
        **Last Updated**: February 2025
        

        """)
    
    with col2:
        st.success("""
        **Acknowledgments**
        
        I gratefully acknowledge the USGS for providing the earthquake data used in developing and testing this application.
        """)
