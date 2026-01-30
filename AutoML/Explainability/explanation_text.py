class ExplanationTextGenerator:
    def generate_local_explanation(self, top_features):
        """
        Convert top features into a human-readable explanation
        """
        if not top_features:
            return "No significant features influenced this prediction."

        feature_names = [feature for feature, _ in top_features]

        if len(feature_names) == 1:
            return f"The prediction was mainly influenced by {feature_names[0]}."

        features_text = ", ".join(feature_names[:-1]) + " and " + feature_names[-1]
        return f"The prediction was mainly influenced by {features_text}."

    def generate_global_explanation(self, top_features):
        """
        Generate a global model-level explanation
        """
        if not top_features:
            return "No dominant features were identified."

        feature_names = [feature for feature, _ in top_features]
        features_text = ", ".join(feature_names)

        return f"Overall, the model relies mostly on the following features: {features_text}."

    def generate_lime_explanation(self, lime_top_features):
        """
        Convert LIME top features (list of (name, weight)) into human-readable text.
        """
        if not lime_top_features:
            return "No LIME explanation was generated."
        parts = [f"{name} ({weight:+.3f})" for name, weight in lime_top_features]
        return "LIME (local alternative): the prediction was mainly driven by: " + "; ".join(parts) + "."
