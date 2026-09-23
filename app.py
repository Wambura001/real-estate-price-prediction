import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set up clean web tab configuration
st.set_page_config(
    page_title="Ames Property Valuation Engine",
    page_icon="🏠",
    layout="centered"
)

# 1. Pipeline Artifact Loader
# 1. Pipeline Artifact Loader
@st.cache_resource
def load_production_pipeline():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.normpath(os.path.join(BASE_DIR, '..', 'models', 'valuation_engine_pipeline.joblib'))
    
    if not os.path.exists(model_path):
        st.error(f"Production model artifact not found at {model_path}. Please run src/deploy.py first!")
        return None
    return joblib.load(model_path)



pipeline = load_production_pipeline()

# 2. Main Title Layout
st.title("Automated Property Valuation Engine")
st.markdown("""
This production dashboard utilizes an optimized statistical learning pipeline to predict the market valuation of residential assets based on physical and structural metrics.
""")
st.write("---")

if pipeline is not None:
    st.subheader("Enter Asset Feature Profile")
    
    # 3. Create interactive input elements for the core value drivers
    col1, col2 = st.columns(2)
    
    with col1:
        overall_qual = st.slider(
            "Overall Material & Finish Quality", 
            min_value=1, max_value=10, value=6,
            help="1: Very Poor, 5: Average, 10: Very Excellent"
        )
        gr_liv_area = st.number_input(
            "Above Ground Living Area (Sq Ft)", 
            min_value=300, max_value=6000, value=1500, step=50
        )
        garage_cars = st.slider(
            "Garage Car Capacity", 
            min_value=0, max_value=4, value=2
        )

    with col2:
        total_bsmt_sf = st.number_input(
            "Total Basement Size (Sq Ft)", 
            min_value=0, max_value=6000, value=1000, step=50
        )
        year_built = st.slider(
            "Year House Was Constructed", 
            min_value=1870, max_value=2010, value=1970
        )
        neighborhood = st.selectbox(
            "Neighborhood Location",
            options=['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'Gilbert', 'NridgHt', 'Sawyer']
        )

    # 4. Synthesize the inference payload
    if st.button("Calculate Estimated Market Value", type="primary"):
        with st.spinner("Processing asset vectors through the inference engine..."):
            
            # Create a blank DataFrame with a single row matching the original training structure
            # We load the raw dataset columns to serve as a baseline template frame
            # Create a blank DataFrame with a single row matching the original training structure
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            template_path = os.path.normpath(os.path.join(BASE_DIR, '..', 'data', 'raw', 'ames_housing_raw.csv'))
            raw_data_template = pd.read_csv(template_path).drop(columns=['SalePrice', 'Id'], errors='ignore')

            
            # Fill the template row with median/mode values from the original data dataframes
            input_payload = pd.DataFrame([raw_data_template.iloc[0].copy()])
            
            # Inject our user's specific web inputs over the template defaults
            input_payload['OverallQual'] = overall_qual
            input_payload['GrLivArea'] = gr_liv_area
            input_payload['GarageCars'] = garage_cars
            input_payload['TotalBsmtSF'] = total_bsmt_sf
            input_payload['YearBuilt'] = year_built
            input_payload['Neighborhood'] = neighborhood
            
            # Drop any columns that were filtered out during deployment cleaning steps
            columns_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType']
            input_payload = input_payload.drop(columns=[col for col in columns_to_drop if col in input_payload.columns])
            
            # 5. Run real-time prediction pipeline query
            log_prediction = pipeline.predict(input_payload)
            final_dollar_price = np.expm1(log_prediction)[0]
            
            # 6. Display results back to user with formatting
            st.write("---")
            st.success(f"### Predicted Market Value: **${final_dollar_price:,.2f}**")
            st.metric(label="Calculated Asset Valuation (USD)", value=f"${final_dollar_price:,.2f}")

st.write("---")
st.caption("**Technical Academic Disclaimer:** This model is calibrated for predictive demonstration exercises using historical real estate distributions and does not substitute for licensed property appraisals.")
