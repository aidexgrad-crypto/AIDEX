import shap
import matplotlib.pyplot as plt
import numpy as np
from lime.lime_tabular import LimeTabularExplainer


class XAIEngine:
    def __init__(self, model, X_train, X_test):
        """
        model: trained ML model
        X_train: training features (DataFrame)
        X_test: test features (DataFrame)
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = None

    def compute_shap_values(self):
        """
        Compute SHAP values for test data
        """
        self.shap_values = self.explainer.shap_values(self.X_test)
        return self.shap_values

    def global_explanation(self):
        """
        Global feature importance (model-level explanation)
        """
        if self.shap_values is None:
            self.compute_shap_values()

        shap.summary_plot(self.shap_values, self.X_test, show=False)
        plt.tight_layout()
        plt.show()

    def save_global_explanation(self, path="shap_global.png"):
        """Save global SHAP summary plot to file (human-readable graph)."""
        if self.shap_values is None:
            self.compute_shap_values()
        shap.summary_plot(self.shap_values, self.X_test, show=False)
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight", dpi=100)
        plt.close()

    def local_explanation(self, index=0):
        """
        Local explanation for a single prediction (SHAP)
        """
        if self.shap_values is None:
            self.compute_shap_values()

        # Binary classification → take class 1
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[1][index]
        else:
            shap_vals = self.shap_values[index]

        shap_vals = np.array(shap_vals).flatten()

        shap.bar_plot(shap_vals, show=False)
        plt.tight_layout()
        plt.show()

    def save_local_explanation(self, index=0, path="shap_local.png"):
        """Save local SHAP bar plot for one prediction to file (human-readable graph)."""
        if self.shap_values is None:
            self.compute_shap_values()
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[1][index]
        else:
            shap_vals = self.shap_values[index]
        shap_vals = np.array(shap_vals).flatten()
        shap.bar_plot(shap_vals, show=False)
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight", dpi=100)
        plt.close()

    def get_top_features(self, index=0, top_n=5):
        """
        Return top contributing SHAP features for one instance
        """
        if self.shap_values is None:
            self.compute_shap_values()

        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[1][index]
        else:
            shap_vals = self.shap_values[index]

        shap_vals = np.array(shap_vals).flatten()
        feature_names = list(self.X_test.columns)

        feature_impact = {
            feature_names[i]: float(shap_vals[i])
            for i in range(len(feature_names))
        }

        sorted_features = sorted(
            feature_impact.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        return sorted_features[:top_n]

    def get_global_top_features(self, top_n=5):
        """
        Return globally most important features (mean |SHAP| across test set).
        """
        if self.shap_values is None:
            self.compute_shap_values()

        if isinstance(self.shap_values, list):
            shap_arr = np.array(self.shap_values[1])
        else:
            shap_arr = np.array(self.shap_values)

        mean_abs = np.abs(shap_arr).mean(axis=0)
        # Flatten so we can index scalars (handles 1D/2D from different SHAP outputs)
        mean_abs_flat = np.asarray(mean_abs).flatten()
        feature_names = list(self.X_test.columns)

        feature_impact = [
            (feature_names[i], float(mean_abs_flat[i]))
            for i in range(len(feature_names))
        ]
        sorted_features = sorted(
            feature_impact,
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:top_n]

    def lime_explanation(self, index=0, num_features=10):
        """
        Generate LIME explanation for one prediction.
        Returns a LIME Explanation object (use .as_list() for feature weights).
        """
        explainer = LimeTabularExplainer(
            training_data=self.X_train.values,
            feature_names=self.X_train.columns.tolist(),
            class_names=["Benign", "Malignant"],
            mode="classification",
        )
        explanation = explainer.explain_instance(
            data_row=self.X_test.values[index],
            predict_fn=self.model.predict_proba,
            num_features=num_features,
        )
        return explanation

    def get_lime_top_features(self, index=0, top_n=5):
        """
        Return top LIME feature weights for one instance: list of (feature_name, weight).
        Used to build human-readable LIME text for the API/chatbot.
        """
        exp = self.lime_explanation(index=index, num_features=top_n)
        # as_list() returns list of (feature_name, weight), already ordered by importance
        return exp.as_list()[:top_n]

    def save_lime_explanation(self, index=0, path="lime_local.png", num_features=10):
        """Save LIME explanation figure for one prediction to file (human-readable graph)."""
        exp = self.lime_explanation(index=index, num_features=num_features)
        fig = exp.as_pyplot_figure()
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight", dpi=100)
        plt.close(fig)


# -------------------------------------------------
# Simple text explanation helper
# -------------------------------------------------
def simple_text_explanation(top_features):
    """
    Convert top SHAP features into simple human-readable text.
    """
    sentences = []

    for feature, value in top_features:
        if value > 0:
            sentences.append(f"{feature} increased the prediction.")
        else:
            sentences.append(f"{feature} decreased the prediction.")

    return sentences
