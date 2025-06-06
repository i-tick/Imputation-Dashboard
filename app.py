from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import BytesIO
import uvicorn
from sklearn.experimental import enable_iterative_imputer  # <-- add this line
from sklearn.impute import IterativeImputer
import numpy as np

app = FastAPI()

# Allow CORS from any origin (for local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def dummy_impute(df: pd.DataFrame, cols: str, max_iterator=25, rand_state=42):

    df_copy = df.copy()
    original_series = df[cols]
    mask = original_series.isna()
    imputer = IterativeImputer(max_iter=max_iterator, random_state=rand_state)

    # Create a list to store values after each iteration
    iteration_values = []

    # Get all numerical columns
    numerical_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    # print('Numerical columns:', numerical_cols)

    # Fit the imputer and transform the data
    imputed_values = imputer.fit_transform(df_copy[numerical_cols])

    # Store the values after each iteration
    for i in range(max_iterator):
        # Get the current state of the imputer
        current_values = imputer.transform(df_copy[numerical_cols])
        iteration_values.append(current_values)

    # Extract only the imputed values for the specified columns
    imputed_cols_values = imputed_values[:, [numerical_cols.index(cols)]]
    # print('Imputed values:', imputed_cols_values)
    
    # Update the DataFrame with the imputed values for the specified columns
    df_copy[cols] = imputed_cols_values[:, 0]
    orig_values = original_series.dropna().tolist()
    imputed_values = df_copy.loc[mask, cols].tolist()

    # print('Imputed',imputed_values)
    # print('Iteration',iteration_values)
    return orig_values, imputed_values

    # original_series = df[column]
    # mask = original_series.isna()
    # filled_value = original_series.mean()
    
    # df_imputed = df.copy()
    # df_imputed[column] = df_imputed[column].fillna(filled_value)

    # orig_values = original_series.dropna().tolist()
    # imputed_values = df_imputed.loc[mask, column].tolist()

    # return orig_values, imputed_values

@app.post("/api/impute")
async def impute_api(
    file: UploadFile = File(...),
    column: str = Form(...),
    iterations: int = Form(...)
):
    contents = await file.read()
    df = pd.read_csv(BytesIO(contents))

    if column not in df.columns:
        return {"error": f"Column '{column}' not found in uploaded CSV."}

    orig_vals, imp_vals = dummy_impute(df, column, iterations)

    return {
        "orig_values": orig_vals,
        "imputed_values": imp_vals
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
