import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import os


class FeatureEngineering:

    def __init__(
        self,
        data_type: str,
        X: pd.DataFrame = None,
        image_root: str = None
    ):
        self.data_type = data_type
        self.X = X
        self.image_root = image_root

    def run(self) -> Dict[str, Any]:
        if self.data_type == "structured":
            return self._engineer_tabular()
        elif self.data_type == "image":
            return self._engineer_image()
        else:
            raise ValueError("Unsupported data type for feature engineering")

    # ================= TABULAR FEATURE ENGINEERING ================= #

    def _engineer_tabular(self) -> Dict[str, Any]:
        engineered_X = self.X.copy()

        numeric_cols = engineered_X.select_dtypes(include=np.number).columns

        # 1️⃣ Polynomial features (numeric only)
        if len(numeric_cols) > 0:
            poly = PolynomialFeatures(
                degree=2, include_bias=False
            )
            poly_features = poly.fit_transform(engineered_X[numeric_cols])

            poly_feature_names = poly.get_feature_names_out(numeric_cols)
            poly_df = pd.DataFrame(
                poly_features, columns=poly_feature_names, index=engineered_X.index
            )

            engineered_X = pd.concat(
                [engineered_X.drop(columns=numeric_cols), poly_df],
                axis=1
            )

        # 2️⃣ Dimensionality reduction (optional safety)
        if engineered_X.shape[1] > 50:
            pca = PCA(n_components=50, random_state=42)
            reduced = pca.fit_transform(engineered_X)

            engineered_X = pd.DataFrame(
                reduced,
                columns=[f"pca_{i}" for i in range(reduced.shape[1])],
                index=engineered_X.index
            )

        return {
            "engineered_features": engineered_X,
            "num_features": engineered_X.shape[1]
        }

    # ================= IMAGE FEATURE ENGINEERING ================= #

    def _engineer_image(self) -> Dict[str, Any]:
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Identity()
        model.eval()

        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        features = []
        image_paths = []

        for root, _, files in os.walk(self.image_root):
            for file in files:
                if file.lower().endswith((".jpg", ".png", ".jpeg")):
                    path = os.path.join(root, file)
                    image_paths.append(path)

                    img = Image.open(path).convert("RGB")
                    tensor = transform(img).unsqueeze(0)

                    with torch.no_grad():
                        embedding = model(tensor).numpy().flatten()

                    features.append(embedding)

        feature_matrix = np.vstack(features)

        return {
            "engineered_features": feature_matrix,
            "num_features": feature_matrix.shape[1],
            "num_samples": feature_matrix.shape[0]
        }
