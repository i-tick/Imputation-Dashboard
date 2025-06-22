import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
import numpy as np
import shap
from kneed import KneeLocator
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

class FeatureImportance:
    def __init__(self, df, target):
        self.df = df.copy()
        self.target = target
        # Encode all non-numerical columns except the target
        non_num_cols = self.df.select_dtypes(exclude=[np.number]).columns.difference([self.target])
        if len(non_num_cols) > 0:
            le = LabelEncoder()
            for col in non_num_cols:
                self.df[col] = le.fit_transform(self.df[col].astype(str))


    def featureCorr(self, corr_type):
        """
        Compute and return sorted feature correlations with the target variable using Pearson or Spearman correlation.
        """
        if corr_type == 'pearson':
            corr_matrix = self.df.corr(method='pearson')
        elif corr_type == 'spearman':
            corr_matrix = self.df.corr(method='spearman')
        else:
            raise ValueError("Invalid correlation type. Use 'pearson' or 'spearman'.")
        
        target_corr = corr_matrix[self.target].drop(self.target)
        sorted_corr = target_corr.reindex(target_corr.abs().sort_values(ascending=False).index)

        return sorted_corr
    
    def featureMutualInfo(self):
        """
        Compute and return the Mutual Information scores for features in df with respect to the target.
        Returns a pandas Series sorted by descending MI score.
        """
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        return mi_series

    def featureRF(self, n_estimators=100, random_state=42):
        """
        Fit a Random Forest regressor and return the feature importances.
        Returns a pandas Series sorted by descending importance.
        """
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        rf.fit(X, y)
        importances = rf.feature_importances_
        rf_series = pd.Series(importances, index=X.columns).sort_values(ascending=False)
        return rf_series

    def featureLasso(self, cv=5, random_state=42):
        """
        Fit a Lasso regression model using cross-validation and return the coefficients.
        Coefficients are sorted by their absolute values in descending order.
        """
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        lasso = LassoCV(cv=cv, max_iter=10000, random_state=random_state)
        lasso.fit(X, y)
        # Create a Series with the coefficients (retain their sign for interpretation)
        lasso_coef = pd.Series(lasso.coef_, index=X.columns)
        # Sort by the absolute value of the coefficients
        lasso_coef_sorted = lasso_coef.reindex(lasso_coef.abs().sort_values(ascending=False).index)
        return lasso_coef_sorted

    def featureRF_SHAP(self, n_estimators=100, random_state=42):
        """
        Fit a Random Forest regressor, compute SHAP values using TreeExplainer,
        and return the mean absolute SHAP values per feature, sorted in descending order.
        """
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        rf.fit(X, y)
        
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X)
        
        # Compute mean absolute SHAP value for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_series = pd.Series(mean_abs_shap, index=X.columns).sort_values(ascending=False)
        return shap_series

    def featureLasso_SHAP(self, cv=5, random_state=42):
        """
        Fit a Lasso regression model, compute SHAP values using LinearExplainer,
        and return the mean absolute SHAP values per feature, sorted in descending order.
        """
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        lasso = LassoCV(cv=cv, random_state=random_state, max_iter=10000)
        lasso.fit(X, y)
        
        explainer = shap.LinearExplainer(lasso, X, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X)
        
        # Compute mean absolute SHAP value for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_series = pd.Series(mean_abs_shap, index=X.columns).sort_values(ascending=False)
        return shap_series


    def impFeatureKnee(self, combined_features, min_combined=0.3):
        sorted_df = combined_features.sort_values(by='Combined', ascending=False)
        x = np.arange(1, len(sorted_df) + 1)
        y = sorted_df['Combined'].values
        
        knee = KneeLocator(x, y, curve='convex', direction='decreasing').knee

        if knee:
            knee_df = sorted_df.head(knee)
        else:
            knee_df = sorted_df.iloc[0:0]  # empty if none found

        thresh_df = sorted_df[sorted_df['Combined'] >= min_combined] #threshold‐based selection

        union_idx = knee_df.index.union(thresh_df.index) #union the two sets
        selected = sorted_df.loc[union_idx]

        signed_imp = selected['Combined'] * selected['Direction_Sign']
        # return list(zip(signed_imp.index.tolist(), signed_imp.tolist()))
        return list(signed_imp.index.tolist())


def plot_feature_importance_heatmap(combined_sorted, knee_features=None):
    """
    Plot a heatmap of feature importances with a knee (elbow) marker.
    combined_sorted: DataFrame with 'Combined' column, index as feature names, sorted descending.
    knee_features: list of feature names selected by the knee/elbow method.
    """
    features = combined_sorted.index.tolist()
    importances = combined_sorted['Combined'].values

    n_features = len(features)
    n_rows = 4  # for visual effect, as in your React grid

    # Create a matrix for the heatmap (repeat importances for n_rows)
    heatmap_data = np.tile(importances, (n_rows, 1))

    fig, ax = plt.subplots(figsize=(max(8, n_features), 3.5))
    im = ax.imshow(heatmap_data, aspect='auto', cmap='Greens', vmin=0, vmax=1)

    # Set feature names as x-ticks
    ax.set_xticks(np.arange(n_features))
    ax.set_xticklabels(features, rotation=90, fontsize=10)
    ax.set_yticks([])  # Hide y-ticks for a cleaner look

    # Axis labels
    ax.set_xlabel('Features', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')

    # Draw knee marker
    if knee_features is not None and len(knee_features) > 0:
        # Find the last index of the knee_features in the sorted features list
        last_knee_idx = max([features.index(f) for f in knee_features if f in features])
        ax.axvline(x=last_knee_idx, color='gray', linestyle='--', linewidth=2)
        # Draw a triangle marker at the top
        ax.plot(last_knee_idx, -0.5, marker='v', color='gray', markersize=12, clip_on=False)

        # Print names of columns after the knee
        after_knee_features = features[last_knee_idx+1:]
        for col in after_knee_features:
            if col in knee_features:
                print(f"Column after knee: {col}")
    # else:
    #     print("Columns after knee:", after_knee_features)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Importance', fontsize=12)

    plt.tight_layout()
    plt.show()