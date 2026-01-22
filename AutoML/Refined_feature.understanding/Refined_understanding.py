import os
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, Any, List
from PIL import Image
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class RefinedFeatureUnderstanding:
    """
    Post-preprocessing Feature Understanding for AIDEX.
    Handles both structured and image data in a unified interface.
    """

    def __init__(
        self,
        data_type: str,
        X: pd.DataFrame = None,
        y: pd.Series = None,
        task_type: str = None,
        image_root: str = None
    ):
        self.data_type = data_type
        self.X = X
        self.y = y
        self.task_type = task_type
        self.image_root = image_root

    def analyze(self) -> Dict[str, Any]:
        if self.data_type == "structured":
            return self._analyze_tabular()
        elif self.data_type == "image":
            return self._analyze_image()
        else:
            raise ValueError("Unsupported data type")

    # ================= TABULAR ================= #

    def _analyze_tabular(self) -> Dict[str, Any]:
        return {
            "low_variance_features": self._low_variance(),
            "correlated_features": self._high_correlation(),
            "feature_importance": self._feature_importance()
        }

    def _low_variance(self, threshold: float = 0.01):
        variances = self.X.var()
        return variances[variances < threshold].index.tolist()

    def _high_correlation(self, threshold: float = 0.9):
        corr = self.X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        return [col for col in upper.columns if any(upper[col] > threshold)]

    def _feature_importance(self):
        if self.task_type == "classification":
            model = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            )
        else:
            model = RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            )

        model.fit(self.X, self.y)
        return dict(
            zip(self.X.columns, model.feature_importances_)
        )

    # ================= IMAGE ================= #

    def _analyze_image(self) -> Dict[str, Any]:
        image_paths = self._collect_images()

        return {
            "num_images": len(image_paths),
            "class_distribution": self._class_distribution(image_paths),
            "image_sizes": self._image_size_stats(image_paths),
            "channel_consistency": self._channel_consistency(image_paths),
            "corrupted_images": self._corrupted_images(image_paths)
        }

    def _collect_images(self) -> List[str]:
        return [
            os.path.join(root, f)
            for root, _, files in os.walk(self.image_root)
            for f in files
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

    def _class_distribution(self, image_paths):
        classes = [os.path.basename(os.path.dirname(p)) for p in image_paths]
        return dict(Counter(classes))

    def _image_size_stats(self, image_paths):
        widths, heights = [], []
        for p in image_paths:
            try:
                with Image.open(p) as img:
                    widths.append(img.width)
                    heights.append(img.height)
            except:
                pass

        return {
            "min_width": min(widths),
            "max_width": max(widths),
            "min_height": min(heights),
            "max_height": max(heights),
        }

    def _channel_consistency(self, image_paths):
        modes = []
        for p in image_paths:
            try:
                with Image.open(p) as img:
                    modes.append(img.mode)
            except:
                pass
        return dict(Counter(modes))

    def _corrupted_images(self, image_paths):
        corrupted = []
        for p in image_paths:
            try:
                with Image.open(p) as img:
                    img.verify()
            except:
                corrupted.append(p)
        return corrupted
