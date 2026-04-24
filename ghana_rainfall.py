import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score


class GhanaRainfallPredictor:
    """
    Rainfall (Rain / No-Rain) classifier for Ghana's ecological zones.

    Reproduces the preprocessing and model configuration from:
        Appiah-Badu et al., "Rainfall Prediction Using Machine Learning
        Algorithms for the Various Ecological Zones of Ghana",
        IEEE Access, Vol. 10, 2022.

    Expected input DataFrame columns:
        Max_Temp     - Daily maximum temperature (degrees C)
        Min_Temp     - Daily minimum temperature (degrees C)
        RH_0600      - Relative humidity at 06:00 (%)
        RH_1500      - Relative humidity at 15:00 (%)
        Sunshine     - Sunshine duration (hours)
        Wind_Speed   - Wind speed (knots)
        Rainfall_Class - Target label: 1 = Rain, 0 = No-Rain
    """

    FEATURE_COLUMNS = [
        "Max_Temp", "Min_Temp", "RH_0600", "RH_1500", "Sunshine", "Wind_Speed",
    ]
    TARGET_COLUMN = "Rainfall_Class"

    def __init__(self, zone_name="Combined"):
        self.zone_name = zone_name
        self.scaler = MinMaxScaler()
        # Hyperparameters match the paper (Section III, p. 5072):
        # "number of weak learners to be 100 and maximum depth of tree to be 16".
        self.models = {
            "Random_Forest": RandomForestClassifier(
                n_estimators=100, max_depth=16, random_state=42
            ),
            "XGBoost": XGBClassifier(
                n_estimators=100,
                max_depth=16,
                eval_metric="logloss",
                random_state=42,
            ),
        }

    def remove_outliers_iqr(self, dataframe):
        """
        Remove outliers from the climatic features using the IQR rule
        (Appiah-Badu et al., eq. 1): IQR = Q3 - Q1, keep rows within
        [Q1 - 1.5*IQR, Q3 + 1.5*IQR]. The target column is preserved
        unchanged because it is a binary class label.
        """
        features = dataframe[self.FEATURE_COLUMNS]
        q1 = features.quantile(0.25)
        q3 = features.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = ~((features < lower) | (features > upper)).any(axis=1)
        return dataframe.loc[mask].reset_index(drop=True)

    def prepare_and_scale_data(self, dataframe, test_size=0.3):
        """
        Apply Min-Max scaling (eq. 2) to the features and split the data.

        The paper evaluates 70:30, 80:20 and 90:10 splits; 70:30 is the
        default here because it is the most frequently reported ratio.
        """
        X = dataframe[self.FEATURE_COLUMNS]
        y = dataframe[self.TARGET_COLUMN]
        X_scaled = self.scaler.fit_transform(X)
        return train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )

    def train_and_test_models(self, X_train, X_test, y_train, y_test):
        """Train each model and return a formatted evaluation report."""
        header = f"\n{'=' * 20} {self.zone_name.upper()} ZONE RESULTS {'=' * 20}\n"
        report_parts = [header]
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            report_parts.append(f"\nModel: {name.replace('_', ' ')}\n")
            report_parts.append(
                f"Overall Accuracy: {accuracy_score(y_test, predictions):.4f}\n"
            )
            report_parts.append(classification_report(y_test, predictions))
        return "".join(report_parts)
