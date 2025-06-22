from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import BytesIO
import uvicorn
from sklearn.experimental import enable_iterative_imputer  # <-- add this line
from sklearn.impute import IterativeImputer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from customLabelEncoder import CustomLabelEncoder
app = FastAPI()
from featureImportance import FeatureImportance
# from mice import MiceImputer
# from bart import BartImputer

# Allow CORS from any origin (for local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

label_encoder = {}

# @app.post("/api/impute")
# async def impute_api(
#     file: UploadFile = File(...),
#     columns: list[str] = Form(...),  # Accepts array of columns to impute
#     iterations: int = Form(...)
# ):
#     contents = await file.read()
#     df = pd.read_csv(BytesIO(contents))
#     print(f"Imputing columns: {columns} with iterations: {iterations}")

#     miceImputer = BartImputer(df, columns, max_iter=iterations)
#     orig_vals, imp_vals = miceImputer.impute()

#     # print({
#     #     "original_values": orig_vals,
#     #     "imputed_values": imp_vals
#     # })

#     return {
#         "orig_values": orig_vals,
#         "imputed_values": imp_vals
#     }

@app.post("/api/missingness_summary")
async def missingness_summary_api(
    file: UploadFile = File(...)
):
    contents = await file.read()
    df = pd.read_csv(BytesIO(contents))
    missing_percent = (df.isnull().mean() * 100).round(2)
    summary = missing_percent.to_dict()
    print(summary)
    return {"missingness_summary": summary}

@app.post("/api/get_dataframe")
async def get_dataframe_api(
    file: UploadFile = File(...)
):
    contents = await file.read()
    df = pd.read_csv(BytesIO(contents))
    print({
        "dataframe": df.head(5).to_dict(orient="records"),
        "columns": df.columns.tolist(),
        "shape": df.shape
    })
    return {
        "dataframe": df.head(5).to_dict(orient="records"),
        "columns": df.columns.tolist(),
        "shape": df.shape
    }

@app.post("/api/configure_datatype")
async def configure_datatype_api(
    file: UploadFile = File(...),
    column: str = Form(...),
    dtype: str = Form(...),
    treat_none_as_category: bool = Form(False),
    custom_encoder: str = Form(None)
):
    global label_encoder
    global global_df
    contents = await file.read()
    df = pd.read_csv(BytesIO(contents))

    if column not in df.columns:
        return {"error": f"Column '{column}' not found in uploaded CSV."}

    try:
        if dtype == 'Categorical':
            le = CustomLabelEncoder(treat_none_as_category=treat_none_as_category) if custom_encoder else LabelEncoder()

            if not custom_encoder:
                df[column].fillna('SPECIFICALLY_MARKED_MISSING_CATEGORY_PREPROCESSING_DATA', inplace=True)

            # For categorical, convert to string first to avoid conversion issues
            df[column] = df[column].astype(str)

            if custom_encoder:
                # Transform with CustomLabelEncoder without replacing NaNs
                df[column] = le.fit_transform(df, column)
            else:
                # Transform with basic LabelEncoder and replace the missing marker with NaN
                df[column] = le.fit_transform(df[column])
                if "SPECIFICALLY_MARKED_MISSING_CATEGORY_PREPROCESSING_DATA" in le.classes_:
                    missing_val = le.transform(["SPECIFICALLY_MARKED_MISSING_CATEGORY_PREPROCESSING_DATA"])[0]
                    df[column].replace(missing_val, np.nan, inplace=True)

            label_encoder[column] = le
        else:
            # For all other types, try to convert to numeric, if fails, leave as is
            try:
                df[column] = pd.to_numeric(df[column], errors="coerce")
            except Exception:
                return {"error": f"Failed to convert column '{column}' to numeric type."}
        
        global_df = df
        print(df.head(5))
        return {
            "message": f"Column '{column}' configured as {dtype}.",
            "dataframe": df.head(5).to_dict(orient="records"),
            "columns": df.columns.tolist(),
            "shape": df.shape
        }


    except Exception as e:
        return {"error": str(e)}
    

@app.post("/api/feature_importance")
async def feature_importance_api(
    target: str = Form(...),
    method: str = Form(...),
    threshold: float = Form(None),
    top_n: int = Form(None)
):
    
    """
    Combine feature importance scores from multiple methods and return final selected features.

    """
    # Compute individual importance scores using the pre-defined functions
    featureImportance = FeatureImportance(global_df, target)
    pearson_scores = featureImportance.featureCorr('pearson')
    spearman_scores = featureImportance.featureCorr('spearman')
    mutual_scores = featureImportance.featureMutualInfo()
    rf_scores = featureImportance.featureRF()
    lasso_scores = featureImportance.featureLasso().abs()  # using absolute values
    rf_shap_scores = featureImportance.featureRF_SHAP()
    lasso_shap_scores = featureImportance.featureLasso_SHAP()
    # Combine all scores into a DataFrame; index should be the feature names.
    importance_df = pd.DataFrame({
        'Pearson': pearson_scores,
        'Spearman': spearman_scores,
        'MutualInfo': mutual_scores,
        'RF': rf_scores,
        'Lasso': lasso_scores,
        'RF_SHAP': rf_shap_scores,
        'Lasso_SHAP': lasso_shap_scores
    })

    res_cols = []
    match method:
        case 'pearson':
            res_cols.append('Pearson')
        case 'spearman':
            res_cols.append('Spearman')
        case 'mutual_info':
            res_cols.append('MutualInfo')
        case 'rf':
            res_cols.append('RF')
        case 'lasso':
            res_cols.append('Lasso')
        case 'rf_shap':
            res_cols.append('RF_SHAP')
        case 'lasso_shap':
            res_cols.append('Lasso_SHAP')
        case 'all_methods':
            res_cols = ['Pearson', 'Spearman', 'MutualInfo', 'RF', 'Lasso', 'RF_SHAP', 'Lasso_SHAP']
    
    magnitude_df = importance_df.copy()

    for col in res_cols:
        if col not in magnitude_df.columns:
            print(f"Warning: Column '{col}' not found in importance DataFrame. Skipping.")
            continue
        magnitude_df[col] = magnitude_df[col].abs()
        # src code had commented out lines that would convert values to absolute, but we keep it for all methods
        # This is to ensure that the magnitude is always positive, which is useful for averaging.
        # for col in ['Pearson', 'Spearman', 'Lasso']:
        #     magnitude_df[col] = magnitude_df[col].abs()

    def _minmax(s):
        if s.max() == s.min(): # Avoid division by zero
            return s  
        return (s - s.min()) / (s.max() - s.min())
    
    magnitude_df = magnitude_df.apply(_minmax, axis=0)
    importance_df['Combined'] = magnitude_df.mean(axis=1)
    
    directional_columns = ['Pearson', 'Spearman', 'Lasso']  # Add other directional columns as needed
    importance_df['Direction'] = importance_df[directional_columns].mean(axis=1)
    importance_df['Direction_Sign'] = importance_df['Direction'].apply(np.sign) #just sign
    
    combined_sorted = importance_df[['Combined', 'Direction_Sign']].sort_values(by='Combined', ascending=False)
    
    if threshold is not None:
        combined_sorted = combined_sorted[combined_sorted > threshold]
    
    if top_n is not None:
        combined_sorted = combined_sorted.head(top_n)
    
    print(importance_df.head(5))
    knee_features = featureImportance.impFeatureKnee(combined_sorted, min_combined=0.3)
    
    return {"Combined_df":combined_sorted, 
            "knee_features": knee_features
            }

    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
