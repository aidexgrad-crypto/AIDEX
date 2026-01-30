class ExplanationFormatter:
    def __init__(self, feature_names):
        """
        feature_names: list of feature names
        """
        self.feature_names = feature_names

    def format_local_explanation(self, shap_values, index=0, top_n=5):
        """
        Extract top contributing features for one prediction
        """
        values = shap_values[index]

        feature_impact = dict(zip(self.feature_names, values))
        sorted_features = sorted(
            feature_impact.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        return sorted_features[:top_n]

    def format_global_explanation(self, shap_values, top_n=5):
        """
        Extract globally important features
        """
        mean_abs_values = abs(shap_values).mean(axis=0)

        feature_impact = dict(zip(self.feature_names, mean_abs_values))
        sorted_features = sorted(
            feature_impact.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_features[:top_n]
