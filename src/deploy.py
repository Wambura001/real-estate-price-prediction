import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def train_and_serialize_production_model():
    print("Commencing Production Pipeline Assembly...")
    
    # Get the exact absolute path of the directory containing THIS deploy.py script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Construct paths relative to the script location (stepping up one level out of 'src')
    data_path = os.path.join(BASE_DIR, '..', 'data', 'raw', 'ames_housing_raw.csv')
    model_output_dir = os.path.join(BASE_DIR, '..', 'models')
    model_output_file = os.path.join(model_output_dir, 'valuation_engine_pipeline.joblib')
    
    # Normalize paths so Windows reads them correctly
    data_path = os.path.normpath(data_path)
    model_output_file = os.path.normpath(model_output_file)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing base training data array at path: {data_path}")
        
    df = pd.read_csv(data_path)
    
    # 2. Replicate Production Level Cleaning Filters
    columns_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'Id']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice']
    
    # Apply global log transform to training targets for production linear alignment
    y_log = np.log1p(y)
    
    # 3. Reconstruct Feature Engineering Matrix Transformers
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # 4. Bind the Top Performing Architecture (Linear Regression)
    production_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # 5. Fit Pipeline Model on ALL Available Historical Training Points
    print("Training final pricing hyperplane across full domain vectors...")
    production_pipeline.fit(X, y_log)
    
    # 6. Serialize and Save Pipeline to Disk
    os.makedirs(os.path.normpath(model_output_dir), exist_ok=True)
    joblib.dump(production_pipeline, model_output_file)
    print(f"Production artifact successfully saved to: {model_output_file}")
    return production_pipeline



def simulate_real_time_inference(trained_pipeline):
    print("\nTesting Real-Time Inference Mock Application...")
    
    # Load raw file just to extract a sample record structure for demonstration
    sample_df = pd.read_csv(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw', 'ames_housing_raw.csv'))).drop(columns=['SalePrice', 'Id'], errors='ignore')
    single_property_query = sample_df.iloc[[0]].copy() # Isolate row 0 as a raw customer input dictionary
    
    # Execution Block for Real-Time Inference
    # The loaded file contains preprocessing embedded within it, eliminating individual manual row cleaning steps
    log_valuation = trained_pipeline.predict(single_property_query)
    dollar_valuation = np.expm1(log_valuation)[0]
    
    print(f"--- Incoming Real Estate Asset Pricing Diagnostic ---")
    print(f"Predicted Market Asset Valuation: ${dollar_valuation:,.2f}")
    print("Inference Process Terminated Successfully.")


if __name__ == "__main__":
    # Execute structural builds when run directly via the terminal environment
    pipeline_artifact = train_and_serialize_production_model()
    simulate_real_time_inference(pipeline_artifact)
    