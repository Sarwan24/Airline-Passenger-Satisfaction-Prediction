import streamlit as st
import joblib

st.set_page_config(
    page_title="Airline Satisfaction Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model
model = joblib.load("Models/airline_satisfaction_model.pkl")


st.sidebar.title("✈️ Airline Satisfaction")

st.sidebar.markdown("---")

st.sidebar.info("""
### 🤖 Model Information

**Algorithm:** Random Forest Classifier

**Accuracy:** 96%

**Developer:** Sarwan Kushwaha

**Framework:** Streamlit

**Library:** Scikit-Learn
""")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Model Performance")

st.sidebar.metric("Accuracy", "96.35%")
st.sidebar.metric("Precision", "97.27%")
st.sidebar.metric("Recall", "94.26%")
st.sidebar.metric("F1 Score", "95.74%")
st.sidebar.metric("ROC-AUC", "96.11%")

st.sidebar.markdown("---")



st.title("✈️ Airline Passenger Satisfaction Prediction")

st.markdown("""
Predict whether a passenger is **Satisfied** or **Neutral / Dissatisfied**
using Machine Learning.
""")

st.markdown("---")


# Page Config
st.set_page_config(
    page_title="Airline Satisfaction Prediction",
    page_icon="✈️",
    layout="wide"
)


# -------------------------
# Passenger Information
# -------------------------

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    customer_type = st.selectbox("Customer Type", ["Returning", "First-time"])
    travel_type = st.selectbox("Type of Travel", ["Business", "Personal"])
    travel_class = st.selectbox("Class", ["Business", "Economy", "Economy Plus"])

with col2:
    age = st.number_input("Age", 5, 100, 30)
    flight_distance = st.number_input("Flight Distance", 0, 10000, 500)
    departure_delay = st.number_input("Departure Delay", 0, 2000, 0)
    arrival_delay = st.number_input("Arrival Delay", 0, 2000, 0)
    
    
    
st.subheader("✈️ Service Ratings")

col3, col4 = st.columns(2)

with col3:
    departure_arrival_time = st.slider("Departure/Arrival Time Convenience", 0, 5, 3)
    online_booking = st.slider("Online Booking", 0, 5, 3)
    checkin_service = st.slider("Check-in Service", 0, 5, 3)
    online_boarding = st.slider("Online Boarding", 0, 5, 3)
    gate_location = st.slider("Gate Location", 0, 5, 3)
    onboard_service = st.slider("On-board Service", 0, 5, 3)
    seat_comfort = st.slider("Seat Comfort", 0, 5, 3)

with col4:
    legroom_service = st.slider("Leg Room Service", 0, 5, 3)
    cleanliness = st.slider("Cleanliness", 0, 5, 3)
    food_drink = st.slider("Food & Drink", 0, 5, 3)
    inflight_service = st.slider("Inflight Service", 0, 5, 3)
    inflight_wifi = st.slider("Inflight WiFi Service", 0, 5, 3)
    inflight_entertainment = st.slider("Inflight Entertainment", 0, 5, 3)
    baggage_handling = st.slider("Baggage Handling", 0, 5, 3)
    

# step 9
    
import pandas as pd

if st.button("🔍 Predict Satisfaction"):

    # Convert categorical values
    gender_male = 1 if gender == "Male" else 0
    customer_returning = 1 if customer_type == "Returning" else 0
    travel_personal = 1 if travel_type == "Personal" else 0

    # Class Encoding
    class_economy = 1 if travel_class == "Economy" else 0
    class_economy_plus = 1 if travel_class == "Economy Plus" else 0

    # Create DataFrame
    input_df = pd.DataFrame([{
        "Age": age,
        "FlightDistance": flight_distance,
        "DepartureDelay": departure_delay,
        "ArrivalDelay": arrival_delay,
        "DepartureArrivalTimeConvenience": departure_arrival_time,
        "OnlineBooking": online_booking,
        "CheckInService": checkin_service,
        "OnlineBoarding": online_boarding,
        "GateLocation": gate_location,
        "OnBoardService": onboard_service,
        "SeatComfort": seat_comfort,
        "LegRoomService": legroom_service,
        "Cleanliness": cleanliness,
        "Food&Drink": food_drink,
        "InflightService": inflight_service,
        "InflightWifiService": inflight_wifi,
        "InflightEntertainment": inflight_entertainment,
        "BaggageHandling": baggage_handling,
        "Gender_Male": gender_male,
        "CustomerType_Returning": customer_returning,
        "TypeTravel_Personal": travel_personal,
        "Class_Economy": class_economy,
        "Class_Economy Plus": class_economy_plus
    }])

    # Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)

    
    # Confidence Score
    confidence = max(probability[0]) * 100

    st.markdown("---")

    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.success("😊 Passenger is Satisfied")
        else:
            st.error("😞 Passenger is Neutral or Dissatisfied")

    with col2:
        st.metric(
        label="Confidence",
        value=f"{confidence:.2f}%"
    )

    st.progress(confidence / 100)
    
    
    st.markdown("---")

    st.subheader("📈 Prediction Probability")

    col1, col2 = st.columns(2)

    satisfied_prob = probability[0][1] * 100
    neutral_prob = probability[0][0] * 100

    with col1:
        st.write("😊 Satisfied")
        st.progress(satisfied_prob / 100)
        st.write(f"{satisfied_prob:.2f}%")

    with col2:
        st.write("😞 Neutral / Dissatisfied")
        st.progress(neutral_prob / 100)
        st.write(f"{neutral_prob:.2f}%")
    
    

st.markdown("---")
st.subheader("🧾 Passenger Summary")

summary = pd.DataFrame({
    "Feature": [
        "Gender",
        "Age",
        "Customer Type",
        "Travel Type",
        "Class",
        "Flight Distance (km)"
    ],
    "Value": [
        gender,
        age,
        customer_type,
        travel_type,
        travel_class,
        flight_distance
    ]
})

st.table(summary)






    
    
st.markdown("---")

st.markdown(
"""
<center>

Made with ❤️ using Python | Streamlit | Scikit-Learn

© 2026 Sarwan Kushwaha

</center>
""",
unsafe_allow_html=True
)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    