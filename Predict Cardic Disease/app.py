import streamlit as st
import pandas as pd
from cardiac_disease_prediction import load_data, preprocess_data, train_and_evaluate_models
from sklearn.model_selection import train_test_split

def main():
    st.title("Heart Disease Prediction System")
    st.write("""
    This application predicts the likelihood of heart disease based on various health parameters.
    Please enter your health information below:
    """)
    
    # Load and prepare the model
    df = load_data()
    X_scaled, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])[1]['model']
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=20, max_value=100, value=55)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=90, max_value=200, value=130)
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=240)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        restecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    
    with col2:
        thalach = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=6.0, value=1.5)
        slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
        ca = st.number_input("Number of Major Vessels", min_value=0, max_value=3, value=0)
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
    
    # Convert categorical inputs to numerical
    sex = 1 if sex == "Male" else 0
    cp = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
    fbs = 1 if fbs == "Yes" else 0
    restecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg)
    exang = 1 if exang == "Yes" else 0
    slope = ["Upsloping", "Flat", "Downsloping"].index(slope)
    thal = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)
    
    # Create input dictionary
    user_input = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    
    if st.button("Predict"):
        # Scale the input
        input_df = pd.DataFrame([user_input])
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = best_model.predict(input_scaled)[0]
        probability = best_model.predict_proba(input_scaled)[0]
        
        # Display result
        st.write("---")
        if prediction == 1:
            st.error("⚠️ High Risk: Heart Disease Predicted")
            st.write(f"Probability of heart disease: {probability[1]:.2%}")
        else:
            st.success("✅ Low Risk: No Heart Disease Predicted")
            st.write(f"Probability of no heart disease: {probability[0]:.2%}")
        
        # Display feature importance if using Random Forest
        if hasattr(best_model, 'feature_importances_'):
            st.write("\n### Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': list(user_input.keys()),
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            st.bar_chart(feature_importance.set_index('Feature'))

if __name__ == "__main__":
    main() 