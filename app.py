import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report

from ghana_rainfall import GhanaRainfallPredictor
from data_generator import ZONES as ZONE_RANGES


ZONE_INFO = {
    "Coastal": {
        "annual_mm": 900,
        "pattern": "Bi-modal",
        "note": "Driest zone; rainfall modulated by the land-sea breeze circulation.",
    },
    "Forest": {
        "annual_mm": 2200,
        "pattern": "Bi-modal",
        "note": "Wettest zone; rain throughout most of the year.",
    },
    "Transition": {
        "annual_mm": 1300,
        "pattern": "Mixed",
        "note": "Sits between Savannah and Forest; shows traits of both climates.",
    },
    "Savannah": {
        "annual_mm": 1100,
        "pattern": "Uni-modal",
        "note": "Warm year-round; a single wet season.",
    },
}


def zone_defaults(zone_name):
    """Mid-range values from data_generator.ZONES for sensible per-zone defaults."""
    cfg = ZONE_RANGES[zone_name.lower()]
    mid = lambda lo_hi: (lo_hi[0] + lo_hi[1]) / 2.0
    return {
        "Max_Temp": round(mid(cfg["max_t"]), 1),
        "Min_Temp": round(mid(cfg["min_t"]), 1),
        "RH_0600": int(round(mid(cfg["rh06"]))),
        "RH_1500": int(round(mid(cfg["rh15"]))),
        "Sunshine": round(mid(cfg["sun"]), 1),
        "Wind_Speed": round(mid(cfg["wind"]), 1),
    }


st.set_page_config(
    page_title="Group 6: Ghana Rainfall Predictor",
    page_icon="🌧️",
    layout="wide",
)


@st.cache_resource(show_spinner="Training model...")
def load_trained_predictor(zone_name):
    """Load zone data, clean, scale, train XGBoost, and capture test metrics."""
    data = pd.read_csv(f"{zone_name.lower()}_zone_data.csv")
    predictor = GhanaRainfallPredictor(zone_name=zone_name)
    clean_df = predictor.remove_outliers_iqr(data)
    X_train, X_test, y_train, y_test = predictor.prepare_and_scale_data(clean_df)

    model = predictor.models["XGBoost"]
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "report": classification_report(
            y_test, preds, target_names=["No Rain", "Rain"], digits=3
        ),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "rain_rate": float(y_train.mean()),
        "features": predictor.FEATURE_COLUMNS,
        "importances": dict(
            zip(predictor.FEATURE_COLUMNS, model.feature_importances_.tolist())
        ),
    }
    return predictor, model, metrics


# --- Sidebar ------------------------------------------------------------
st.sidebar.title("Settings")
zone = st.sidebar.selectbox(
    "Ecological Zone", list(ZONE_INFO.keys()), index=1  # default Forest
)

info = ZONE_INFO[zone]
st.sidebar.markdown(
    f"**Mean annual rainfall:** ~{info['annual_mm']:,} mm  \n"
    f"**Pattern:** {info['pattern']}"
)
st.sidebar.caption(info["note"])
st.sidebar.divider()

# Reset form values whenever the selected zone changes.
if st.session_state.get("_active_zone") != zone:
    st.session_state["_active_zone"] = zone
    for k, v in zone_defaults(zone).items():
        st.session_state[f"in_{k}"] = v

if st.sidebar.button("Reset to zone defaults", use_container_width=True):
    for k, v in zone_defaults(zone).items():
        st.session_state[f"in_{k}"] = v
    st.rerun()

# --- Load / train model -------------------------------------------------
try:
    predictor_tool, trained_model, metrics = load_trained_predictor(zone)
except FileNotFoundError:
    st.error(
        "Zone data files not found. Run `python data_generator.py` first "
        "to create the zone CSVs, then reload this page."
    )
    st.stop()

# --- Header -------------------------------------------------------------
st.title("🌧️ Rainfall Prediction System")
st.caption(
    "Rain / No-Rain classifier for Ghana's four ecological zones, following "
    "Appiah-Badu et al. (2022), IEEE Access."
)

tab_predict, tab_performance, tab_about = st.tabs(
    ["Predict", "Model Performance", "About"]
)

# =======================================================================
# Tab 1: Predict
# =======================================================================
with tab_predict:
    st.subheader(f"Weather Parameters — {zone} Zone")

    with st.form("prediction_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            st.number_input(
                "Max Temp (°C)", min_value=10.0, max_value=50.0, step=0.5,
                key="in_Max_Temp",
            )
            st.number_input(
                "Min Temp (°C)", min_value=10.0, max_value=40.0, step=0.5,
                key="in_Min_Temp",
            )
            st.slider(
                "Humidity @ 6AM (%)", 0, 100, key="in_RH_0600",
            )
        with col2:
            st.number_input(
                "Sunshine (Hours)", min_value=0.0, max_value=14.0, step=0.5,
                key="in_Sunshine",
            )
            st.number_input(
                "Wind Speed (Knots)", min_value=0.0, max_value=30.0, step=0.5,
                key="in_Wind_Speed",
            )
            st.slider(
                "Humidity @ 3PM (%)", 0, 100, key="in_RH_1500",
            )

        submit = st.form_submit_button(
            "Predict Rainfall", type="primary", use_container_width=True
        )

    if submit:
        input_df = pd.DataFrame(
            [{
                "Max_Temp": st.session_state["in_Max_Temp"],
                "Min_Temp": st.session_state["in_Min_Temp"],
                "RH_0600": st.session_state["in_RH_0600"],
                "RH_1500": st.session_state["in_RH_1500"],
                "Sunshine": st.session_state["in_Sunshine"],
                "Wind_Speed": st.session_state["in_Wind_Speed"],
            }],
            columns=GhanaRainfallPredictor.FEATURE_COLUMNS,
        )
        input_scaled = predictor_tool.scaler.transform(input_df)
        prediction = int(trained_model.predict(input_scaled)[0])
        p_rain = float(trained_model.predict_proba(input_scaled)[0][1])
        p_no_rain = 1.0 - p_rain

        st.divider()
        st.subheader("Prediction")

        if prediction == 1:
            st.success(
                f"### 🌧️ RAIN expected  —  confidence {p_rain * 100:.1f}%"
            )
        else:
            st.warning(
                f"### ☀️ NO RAIN expected  —  confidence {p_no_rain * 100:.1f}%"
            )

        m1, m2 = st.columns(2)
        with m1:
            st.metric("P(Rain)", f"{p_rain * 100:.1f}%")
            st.progress(p_rain)
        with m2:
            st.metric("P(No Rain)", f"{p_no_rain * 100:.1f}%")
            st.progress(p_no_rain)

        st.caption(
            f"Model: XGBoost trained on {metrics['n_train']:,} {zone} samples "
            f"(test accuracy {metrics['accuracy'] * 100:.2f}%)."
        )

# =======================================================================
# Tab 2: Model Performance
# =======================================================================
with tab_performance:
    st.subheader(f"XGBoost Performance — {zone} Zone")
    st.caption(
        "Metrics are computed on a held-out 30% test split (stratified, "
        "random_state=42) after IQR outlier removal and Min-Max scaling."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    c2.metric("Training Samples", f"{metrics['n_train']:,}")
    c3.metric("Test Samples", f"{metrics['n_test']:,}")
    c4.metric("Base Rain Rate", f"{metrics['rain_rate'] * 100:.1f}%")

    st.markdown("#### Feature Importance")
    importance_df = (
        pd.DataFrame(
            {"Feature": list(metrics["importances"].keys()),
             "Importance": list(metrics["importances"].values())}
        )
        .sort_values("Importance", ascending=True)
        .set_index("Feature")
    )
    st.bar_chart(importance_df, horizontal=True)

    with st.expander("Classification report (precision, recall, F1 per class)"):
        st.code(metrics["report"], language="text")

# =======================================================================
# Tab 3: About
# =======================================================================
with tab_about:
    st.subheader("About This Project")
    st.markdown(
        """
        This application reproduces, in a runnable form, the methodology of:

        > **Appiah-Badu, N.K.A., Missah, Y.M., Amekudzi, L.K., Ussiph, N.,
        > Frimpong, T., Ahene, E.** (2022). *Rainfall Prediction Using
        > Machine Learning Algorithms for the Various Ecological Zones of
        > Ghana.* IEEE Access, Vol. 10.

        **Pipeline:**
        1. **IQR outlier removal** on the six climatic features
           (target class excluded).
        2. **Min-Max scaling** of features to `[0, 1]`.
        3. **Stratified 70:30 train/test split**.
        4. **XGBoost** (`n_estimators=100`, `max_depth=16`) — matches the
           paper's top-performing configuration.

        **Features used:** Max Temp, Min Temp, Humidity at 06:00 and 15:00,
        Sunshine hours, Wind speed.

        **Note on data:** The original GMet 40-year dataset is not
        redistributable. The CSVs here are synthetic: feature ranges follow
        the paper's zone descriptions, and labels are drawn from a
        logistic function of humidity, sunshine and temperature so the
        classifier has real signal to learn. Accuracy on synthetic data is
        **not** directly comparable to the paper's published figures.
        """
    )

    st.markdown("#### Ecological Zones at a Glance")
    zone_table = pd.DataFrame(
        [
            {"Zone": z, "Annual Rainfall (mm)": v["annual_mm"],
             "Pattern": v["pattern"], "Notes": v["note"]}
            for z, v in ZONE_INFO.items()
        ]
    )
    st.dataframe(zone_table, hide_index=True, use_container_width=True)
