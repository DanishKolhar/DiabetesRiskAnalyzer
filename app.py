import glob
import os
import pickle
import joblib
import streamlit as st
import pandas as pd
from pathlib import Path


def find_model_and_scaler():
    # Search for model.pkl and scaler.pkl in this folder and parent (project root),
    # then fall back to a recursive glob from the current working directory.
    model_path = None
    scaler_path = None

    # check frontend and project root
    search_dirs = [Path(__file__).parent, Path(__file__).parent.parent]
    for d in search_dirs:
        if not d.exists():
            continue
        mc = list(d.rglob("model.pkl"))
        sc = list(d.rglob("scaler.pkl"))
        if mc and model_path is None:
            model_path = str(mc[0])
        if sc and scaler_path is None:
            scaler_path = str(sc[0])
        if model_path and scaler_path:
            break

    # fallback: search recursively from cwd
    if not model_path:
        mc = glob.glob("**/model.pkl", recursive=True)
        model_path = mc[0] if mc else None
    if not scaler_path:
        sc = glob.glob("**/scaler.pkl", recursive=True)
        scaler_path = sc[0] if sc else None

    return model_path, scaler_path


def load_pickle(path):
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺")

st.title("🩺 Diabetes Risk Predictor")
st.write("Provide patient information and click Evaluate to get the model prediction.")

model_path, scaler_path = find_model_and_scaler()
if not model_path:
    st.error("No `model.pkl` found in the workspace. Place `model.pkl` in the project root or a subfolder.")
    st.stop()



model = None
scaler = None
with st.spinner("Loading model and scaler (if available)..."):
    try:
        model = load_pickle(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    if scaler_path:
        try:
            scaler = load_pickle(scaler_path)
           
        except Exception as e:
            st.warning(f"Found scaler but failed to load: {e}")


st.divider()

# ---------------------------
# Sidebar – Patient Profile
# ---------------------------
st.sidebar.header("🧑‍⚕️ Patient Profile")

Sex = st.sidebar.selectbox("🧬 Sex", ["Female", "Male"])
Sex = 0 if Sex == "Female" else 1

Age = st.sidebar.slider("🎂 Age Category (1–13)", 1, 13, 7)
Education = st.sidebar.slider("🎓 Education Level (1–6)", 1, 6, 3)
Income = st.sidebar.slider("💰 Income Category (1–8)", 1, 8, 4)

st.sidebar.divider()

st.sidebar.header("🏃 Lifestyle & Habits")

Smoker = st.sidebar.selectbox("🚬 Smoker", [0, 1])
PhysActivity = st.sidebar.selectbox("🏃 Physically Active", [0, 1])
HvyAlcoholConsump = st.sidebar.selectbox("🍺 Heavy Alcohol Use", [0, 1])

Fruits = st.sidebar.selectbox("🍎 Fruits Daily", [0, 1])
Veggies = st.sidebar.selectbox("🥦 Vegetables Daily", [0, 1])

# ---------------------------
# Main – Clinical Inputs
# ---------------------------
st.subheader("🩸 Clinical Measurements")

col1, col2 = st.columns(2)

with col1:
    Sugar = st.number_input("🩸 Blood Glucose (mg/dL)", 60.0, 400.0, 110.0)
    HighBP = st.selectbox("🫀 High Blood Pressure", [0, 1])
    HighChol = st.selectbox("🧪 High Cholesterol", [0, 1])
    CholCheck = st.selectbox("🔬 Cholesterol Checked (Last 5 Years)", [0, 1])

with col2:
    height = st.number_input("📏 Height (meters)", 1.0, 2.5, 1.7)
    weight = st.number_input("⚖️ Weight (kg)", 30.0, 200.0, 65.0)

    BMI = weight / (height ** 2)
    st.metric("🧮 Body Mass Index (BMI)", f"{BMI:.2f}")

    if BMI < 18.5:
        st.info("🟦 BMI Category: Underweight")
    elif BMI < 25:
        st.success("🟩 BMI Category: Normal")
    elif BMI < 30:
        st.warning("🟨 BMI Category: Overweight")
    else:
        st.error("🟥 BMI Category: Obese")

st.divider()

# ---------------------------
# Medical History
# ---------------------------
st.subheader("🧠 Medical History")

col3, col4 = st.columns(2)

with col3:
    Stroke = st.selectbox("🧠 History of Stroke", [0, 1])
    HeartDiseaseorAttack = st.selectbox("❤️ Heart Disease / Attack", [0, 1])
    DiffWalk = st.selectbox("🦵 Difficulty Walking", [0, 1])

with col4:
    AnyHealthcare = st.selectbox("🏥 Healthcare Coverage", [0, 1])
    NoDocbcCost = st.selectbox("💸 Avoided Doctor Due to Cost", [0, 1])

GenHlth = st.slider("🩺 General Health (1 = Excellent, 5 = Poor)", 1, 5, 3)
MentHlth = st.slider("🧠 Poor Mental Health Days (Last 30 Days)", 0, 30, 0)
PhysHlth = st.slider("🦴 Poor Physical Health Days (Last 30 Days)", 0, 30, 0)

st.divider()

# ---------------------------
# Prediction Section
# ---------------------------
st.subheader("🔍 Risk Evaluation")

def build_feature_row():
    return {
        "Sugar": Sugar,
        "HighBP": HighBP,
        "HighChol": HighChol,
        "CholCheck": CholCheck,
        "BMI": BMI,
        "Smoker": Smoker,
        "Stroke": Stroke,
        "HeartDiseaseorAttack": HeartDiseaseorAttack,
        "PhysActivity": PhysActivity,
        "Fruits": Fruits,
        "Veggies": Veggies,
        "HvyAlcoholConsump": HvyAlcoholConsump,
        "AnyHealthcare": AnyHealthcare,
        "NoDocbcCost": NoDocbcCost,
        "GenHlth": GenHlth,
        "MentHlth": MentHlth,
        "PhysHlth": PhysHlth,
        "DiffWalk": DiffWalk,
        "Sex": Sex,
        "Age": Age,
        "Education": Education,
        "Income": Income,
    }


if st.button("🧪 Evaluate Diabetes Risk", use_container_width=True):
    features = build_feature_row()
    X = pd.DataFrame([features])

    try:
        # align feature ordering to what the model expects if possible
        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
            missing = [c for c in expected if c not in X.columns]
            if missing:
                st.warning(f"Model expects columns missing from input: {missing}")
            X = X.reindex(columns=expected, fill_value=0)

        X_input = X.values

        # apply scaler if available
        if scaler is not None:
            try:
                X_input = scaler.transform(X_input)
            except Exception as e:
                st.warning(f"Scaler transform failed: {e} — passing raw features to model")

        pred = model.predict(X_input)
        pred_label = int(pred[0]) if hasattr(pred, "__len__") else int(pred)

        prob_text = ""
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X_input)[0]
                prob_text = f" (conf: {max(probs):.2f})"
            except Exception:
                prob_text = ""

        if pred_label == 0:
            st.success(f"🟢 Low Risk: Not Diabetic{prob_text}")
        elif pred_label == 1:
            st.warning(f"🟠 Moderate Risk: Pre-Diabetic{prob_text}")
        else:
            st.error(f"🔴 High Risk: Diabetic{prob_text}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()

st.caption(
    "⚠️ This application is intended for educational and screening purposes only. "
    "It is not a substitute for professional medical diagnosis."
)
