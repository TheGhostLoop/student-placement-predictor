import streamlit as st
import joblib
import pandas as pd

# ------------------- Page -------------------

st.title("🎓 Placement Predictor")
st.info(
    "This application predicts whether a student is likely to be placed based on academic performance, skills, and experience."
)
st.header("Predict Student Placement using Machine Learning")
st.write("Enter the candidate details below.")

# ------------------- Sidebar -------------------

st.sidebar.title("🎓 Placement Predictor")

st.sidebar.markdown("### 🤖 Best Model")
st.sidebar.success("Gradient Boosting Classifier")

st.sidebar.markdown("### 🛠 Tech Stack")

st.sidebar.write("""
- Python
- Scikit-learn
- Streamlit
- Pandas
- Joblib
""")

st.sidebar.markdown("### 📊 Dataset")

st.sidebar.write("""
Student Placement Dataset
(10,051 Records)
""")

st.sidebar.markdown("### 📌 Features")

st.sidebar.write("""
- 13 Input Features
- Binary Classification
- Probability Prediction
""")

st.sidebar.markdown("---")

st.sidebar.caption("Developed by Prince Verma")

# ------------------- Inputs -------------------

age = st.number_input("Enter Age",17,100,step=1)

gender = st.selectbox(
    "Select Gender",
    ["Male","Female"]
)

degree = st.selectbox(
    "Select Degree",
    ["B.Tech","BCA","MCA","B.Sc"]
)

branch = st.selectbox(
    "Select Branch",
    ["CSE","IT","ECE","Civil","ME"]
)

cgpa = st.number_input(
    "CGPA",
    0.0,
    10.0,
    step=0.01,
    format="%.2f"
)

internships = st.slider(
    "Internships",
    0,
    4,
    1
)

projects = st.slider(
    "Projects",
    0,
    6
)

coding_skills = st.slider(
    "Coding Skills",
    1,
    10
)

communication_skills = st.slider(
    "Communication Skills",
    1,
    10
)

aptitude = st.slider(
    "Aptitude Test Score",
    0,
    100
)

soft_skills = st.slider(
    "Soft Skills Rating",
    1,
    10
)

certifications = st.slider(
    "Certifications",
    0,
    4
)

backlogs = st.slider(
    "Backlogs",
    0,
    4
)

# ------------------- Load Model -------------------

@st.cache_resource
def load_model():
    return joblib.load("placement_model.pkl")

model = load_model()

# ------------------- Data -------------------

new_data = pd.DataFrame({

    "Age":[age],
    "Gender":[gender],
    "Degree":[degree],
    "Branch":[branch],
    "CGPA":[cgpa],
    "Internships":[internships],
    "Projects":[projects],
    "Coding_Skills":[coding_skills],
    "Communication_Skills":[communication_skills],
    "Aptitude_Test_Score":[aptitude],
    "Soft_Skills_Rating":[soft_skills],
    "Certifications":[certifications],
    "Backlogs":[backlogs]

})

# ------------------- Prediction -------------------

if st.button("Predict Placement",width='content'):

    prediction = model.predict(new_data)[0]
    probability = model.predict_proba(new_data)[0]

    placed_prob = probability[1]*100
    not_placed_prob = probability[0]*100

    st.divider()

    st.subheader("Prediction Result")

    if prediction==1:

        st.success("✅ Student is likely to be Placed")

        st.progress(placed_prob/100)

        st.metric(
            "Placement Probability",
            f"{placed_prob:.2f}%"
        )

    else:

        st.error("❌ Student is likely to be Not Placed")

        st.progress(not_placed_prob/100)

        st.metric(
            "Not Placement Probability",
            f"{not_placed_prob:.2f}%"
        )

    st.write(f"**Placement Probability:** {placed_prob:.2f}%")
    st.write(f"**Not Placement Probability:** {not_placed_prob:.2f}%")

    # ------------------- Metrics -------------------

    st.divider()

    st.subheader("Model Performance")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Accuracy","100%")
    col2.metric("Precision","100%")
    col3.metric("Recall","100%")
    col4.metric("F1 Score","100%")

    # ------------------- Graph 1 -------------------

    st.divider()

    st.subheader("Model Comparison")

    st.image(
        "accuracy_comparison.png",
        width='content'
    )

    st.caption(
        "Comparison of different Machine Learning algorithms on the same dataset."
    )

    # ------------------- Graph 2 -------------------

    st.subheader("Feature Importance")

    st.image(
        "feature_importances.png",
        width='content'
    )

    st.caption(
        "Communication Skills, Coding Skills, CGPA and Backlogs had the greatest impact on prediction."
    )

    # ------------------- Graph 3 -------------------

    st.subheader("Confusion Matrix")

    st.image(
        "confusion_matrix.png",
        width='content'
    )

    st.caption(
        "The confusion matrix shows the classification performance of the best-performing model."
    )

    # ------------------- About Model -------------------

    st.divider()

    with st.expander("About this Model"):

        st.write("""
### Algorithm
Gradient Boosting Classifier

### Preprocessing
- One Hot Encoding
- ColumnTransformer
- Train/Test Split (80:20)

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score

### Libraries Used
- Scikit-learn
- Pandas
- Streamlit
""")

st.divider()
st.caption("Github:: https://github.com/TheGhostLoop")