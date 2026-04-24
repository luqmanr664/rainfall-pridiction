# Methodology and Technical Specifications
### **Group 6 | Technical Implementation Details**

This document tracks how our implementation maps onto the methodology of:

> Appiah-Badu, N.K.A., Missah, Y.M., Amekudzi, L.K., Ussiph, N., Frimpong, T.,
> Ahene, E. (2022). *Rainfall Prediction Using Machine Learning Algorithms
> for the Various Ecological Zones of Ghana.* **IEEE Access, Vol. 10.**
> DOI: 10.1109/ACCESS.2021.3139312.

---

## 1. Dataset

The paper uses 40 years (1980–2019) of daily observations from **22 synoptic
stations** operated by the Ghana Meteorological Agency (GMet), grouped into
four agro-ecological zones:

| Zone       | Mean annual rainfall | Rainfall pattern |
|------------|----------------------|-------------------|
| Coastal    | ~900 mm              | Bi-modal          |
| Savannah   | ~1100 mm             | Uni-modal, warm   |
| Transition | ~1300 mm             | Mixed             |
| Forest     | ~2200 mm             | Bi-modal, wettest |

The original GMet records are not publicly redistributable, so
`data_generator.py` produces synthetic CSVs whose feature ranges and
base rain rates reflect the paper's descriptive statistics. Labels are
drawn from a logistic function of humidity, sunshine and temperature so
the classifier has genuine signal to learn.

## 2. Features and Target

**Features** (6 climatic variables from the paper's Table 1):

| Column       | Description                           | Unit |
|--------------|---------------------------------------|------|
| `Max_Temp`   | Daily maximum temperature             | °C   |
| `Min_Temp`   | Daily minimum temperature             | °C   |
| `RH_0600`    | Relative humidity at 06:00            | %    |
| `RH_1500`    | Relative humidity at 15:00            | %    |
| `Sunshine`   | Sunshine duration                     | hrs  |
| `Wind_Speed` | Wind speed                            | knots |

**Target**: `Rainfall_Class` — `1` = Rain, `0` = No-Rain.

## 3. Preprocessing Pipeline

The paper's full pipeline (Section III) is:

1. **MICE imputation** of missing values.
2. **IQR outlier removal** on the continuous features.
3. **Oversampling** of the minority class to address class imbalance.
4. **Min-Max scaling** of the features to `[0, 1]`.
5. **Train/test split** at 70:30, 80:20 or 90:10.

Our implementation covers steps 2, 4 and 5. Steps 1 and 3 are omitted
because the synthetic data contains no missing values and is already
approximately balanced per zone (see console output from
`data_generator.py`). If real GMet data is provided, MICE (via
`sklearn.experimental.IterativeImputer`) and SMOTE (via `imblearn`)
should be added before scaling.

### 3.1 IQR Outlier Removal
Formula (paper eq. 1):

$$ IQR = Q3 - Q1 $$

A row is kept when every feature lies in `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]`.
The target column is **excluded** from this filter because IQR on a
binary label is not meaningful.

### 3.2 Min-Max Scaling
Formula (paper eq. 2):

$$ z = \frac{x - \min(x)}{\max(x) - \min(x)} $$

Applied to the features only; the scaler is fitted on the training
partition and re-used at inference time.

### 3.3 Train/Test Split
We use **70:30** with `stratify=y` and `random_state=42`. The paper
additionally evaluates 80:20 and 90:10; these can be reproduced by
passing `test_size` to `prepare_and_scale_data`.

## 4. Machine Learning Models

The paper evaluates five classifiers (DT, RF, MLP, XGB, KNN) and reports
**Random Forest and XGBoost** as the consistent top performers across
zones. Our module implements those two:

| Model          | Configuration (per paper, p. 5072)           |
|----------------|-----------------------------------------------|
| Random Forest  | `n_estimators=100`, `max_depth=16`            |
| XGBoost        | `n_estimators=100`, `max_depth=16`, `eval_metric='logloss'` |

## 5. Evaluation Metrics

Per the paper (Section IV), models are compared on:

- **Accuracy** — overall correct-prediction rate.
- **Precision, Recall, F1-score** — per class (rain / no-rain).

These are produced by `sklearn.metrics.classification_report` in
`train_and_test_models`.

## 6. Simplifications vs. the Paper

| Aspect              | Paper                         | This implementation              |
|---------------------|-------------------------------|----------------------------------|
| Data source         | GMet, 22 stations, 1980–2019  | Synthetic, feature-correlated    |
| Missing values      | MICE imputation               | Not needed (synthetic)           |
| Class imbalance     | Oversampling of minority      | Approximately balanced per zone  |
| Classifiers         | DT, RF, MLP, XGB, KNN         | RF and XGB (top performers)      |
| Split ratios tested | 70:30, 80:20, 90:10           | 70:30 (default, configurable)    |
