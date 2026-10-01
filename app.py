import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

#intialize history
if "history" not in st.session_state:
    st.session_state.history = []

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Sidebar
with st.sidebar:
    st.header("🎓 Project Information")

    st.write("### About")
    st.write(
        "This project uses Machine Learning to predict "
        "student academic performance."
    )

    st.write("### Technologies")
    st.write("🐍 Python")
    st.write("🐼 Pandas")
    st.write("🔢 NumPy")
    st.write("🤖 Scikit-learn")
    st.write("🌐 Streamlit")

    st.write("### Machine Learning Model")
    st.write("Random Forest Regression")

    st.write("### Target")
    st.write("Final Grade (G3)")

st.write(
    "Predict a student's final academic grade using "
    "study time, absences, and previous grades."
)

st.divider()

st.subheader("📋 Student Information")
# Load dataset
data = pd.read_csv("student-mat.csv", sep=";")

# Select features
X = data[["studytime", "absences", "G1", "G2"]]
y = data["G3"]

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,
    random_state=42
)

model.fit(X, y)

# Student inputs
# Student inputs

col1, col2 = st.columns(2)

with col1:
    studytime = st.selectbox(
        "📚 Study Time",
        [1, 2, 3, 4],
        format_func=lambda x: {
            1: "< 2 hours",
            2: "2–5 hours",
            3: "5–10 hours",
            4: "> 10 hours"
        }[x]
    )

    g1 = st.number_input(
        "📊 First Period Grade (G1)",
        min_value=0,
        max_value=20,
        value=10,
        step=1
    )

with col2:
    absences = st.number_input(
        "❌ Number of Absences",
        min_value=0,
        max_value=100,
        value=5,
        step=1
    )

    g2 = st.number_input(
        "📊 Second Period Grade (G2)",
        min_value=0,
        max_value=20,
        value=10,
        step=1
    )

# Prediction button
if st.button("🔮 Predict Final Grade", use_container_width=True):

    input_data = pd.DataFrame(
        [[studytime, absences, g1, g2]],
        columns=["studytime", "absences", "G1", "G2"]
    )

    prediction = model.predict(input_data)[0]

    prediction = max(0, min(20, prediction))

    # Save prediction
    st.session_state.history.append({
        "Study Time": studytime,
        "Absences": absences,
        "G1": g1,
        "G2": g2,
        "Predicted G3": round(prediction, 2)
    })

    st.divider()

    st.subheader("📈 Prediction Result")

    st.metric(
        label="Predicted Final Grade",
        value=f"{prediction:.2f} / 20"
    )

    if prediction >= 10:
        st.success("✅ The predicted grade is passing.")
    else:
        st.warning("⚠️ The predicted grade is below the passing mark.")

# Prediction history
if st.session_state.history:

    st.divider()

    st.subheader("📋 Prediction History")

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True
    )

st.write(
    "This system uses a Random Forest Regression machine learning "
    "model to predict a student's final grade."
)

st.write("The prediction is based on:")

st.write("• Study Time")
st.write("• Number of Absences")
st.write("• First Period Grade (G1)")
st.write("• Second Period Grade (G2)")

st.caption(
    "Dataset: UCI Student Performance Dataset | "
    "Target variable: G3 (Final Grade)"
)
st.divider()

st.caption(
    "🎓 Student Performance Prediction System | "
    "Machine Learning Project"
)