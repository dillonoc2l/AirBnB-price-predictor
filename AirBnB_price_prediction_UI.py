import streamlit as st
import numpy as np
import joblib
import pandas as pd
import numpy as np



#LAUNCH APP WITH 'streamlit run AirBnB_price_prediction_UI.py' IN TERMINAL



# Load model with joblib
model = joblib.load("best_airbnb_model.pkl")


#Formula to calculate distance between two points on a sphere (earth)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

#feature setup
neighbourhoods = [
    'Barking and Dagenham', 'Barnet', 'Bexley', 'Brent', 'Bromley', 'Camden',
    'City of London', 'Croydon', 'Ealing', 'Enfield', 'Greenwich', 'Hackney',
    'Hammersmith and Fulham', 'Haringey', 'Harrow', 'Havering', 'Hillingdon',
    'Hounslow', 'Islington', 'Kensington and Chelsea',
    'Kingston upon Thames', 'Lambeth', 'Lewisham', 'Merton', 'Newham',
    'Redbridge', 'Richmond upon Thames', 'Southwark', 'Sutton',
    'Tower Hamlets', 'Waltham Forest', 'Wandsworth', 'Westminster'
]

room_types = [
    'Entire home/apt',
    'Hotel room',
    'Private room',
    'Shared room'
]



#streamlit UI
st.title("London Airbnb Price Predictor")

st.write("Enter Airbnb details below:")

# Numeric inputs
latitude = st.number_input("Latitude", format="%.6f", value=51.509865)
longitude = st.number_input("Longitude", format="%.6f", value=-0.118092)
minimum_nights = st.number_input("Minimum Nights", min_value=1, value=2)
number_of_reviews = st.number_input("Number of Reviews", min_value=0, value=10)
reviews_per_month = st.number_input("Reviews per Month", min_value=0.0, value=1.0)
calculated_host_listings_count = st.number_input("Host Listings Count", min_value=0, value=1)
availability_365 = st.number_input("Availability (days per year)", min_value=0, max_value=365, value=180)
number_of_reviews_ltm = st.number_input("Reviews Last 12 Months", min_value=0, value=5)

# Dropdowns
selected_neighbourhood = st.selectbox("Neighbourhood", neighbourhoods)
selected_room_type = st.selectbox("Room Type", room_types)

# Numeric features
numeric_features = [
    'dist_to_center', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
    'calculated_host_listings_count', 'availability_365', 'number_of_reviews_ltm'
]

# One-hot encoded features
neighbourhood_features = [f"neighbourhood_{n}" for n in neighbourhoods]
room_type_features = [f"room_type_{r}" for r in room_types]

# Final model features in order used during training
model_features = numeric_features + neighbourhood_features + room_type_features


# bulid input vector
if st.button("Predict Price"):

    # Calculate distance to London center
    center_lat, center_lon = 51.509865, -0.118092
    dist_to_center = haversine(latitude, longitude, center_lat, center_lon)

    # Build feature dictionary initialized to 0
    data = {
        'minimum_nights': minimum_nights,
        'number_of_reviews': number_of_reviews,
        'reviews_per_month': reviews_per_month,
        'calculated_host_listings_count': calculated_host_listings_count,
        'availability_365': availability_365,
        'number_of_reviews_ltm': number_of_reviews_ltm,
        **{f'neighbourhood_{n}': 0 for n in neighbourhoods},      # One-hot encoded neighbourhoods
        **{f'room_type_{r}': 0 for r in room_types},    # One-hot encoded room types
        'dist_to_center': dist_to_center
    }

    # Set the selected neighbourhood and room type to 1
    data[f'neighbourhood_{selected_neighbourhood}'] = 1
    data[f'room_type_{selected_room_type}'] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([data])

    # Reorder columns exactly as in model training
    model_features = [
        'minimum_nights', 'number_of_reviews', 'reviews_per_month',
        'calculated_host_listings_count', 'availability_365',
        'number_of_reviews_ltm',
        'neighbourhood_Barking and Dagenham', 'neighbourhood_Barnet', 'neighbourhood_Bexley',
        'neighbourhood_Brent', 'neighbourhood_Bromley', 'neighbourhood_Camden',
        'neighbourhood_City of London', 'neighbourhood_Croydon', 'neighbourhood_Ealing',
        'neighbourhood_Enfield', 'neighbourhood_Greenwich', 'neighbourhood_Hackney',
        'neighbourhood_Hammersmith and Fulham', 'neighbourhood_Haringey', 'neighbourhood_Harrow',
        'neighbourhood_Havering', 'neighbourhood_Hillingdon', 'neighbourhood_Hounslow',
        'neighbourhood_Islington', 'neighbourhood_Kensington and Chelsea',
        'neighbourhood_Kingston upon Thames', 'neighbourhood_Lambeth', 'neighbourhood_Lewisham',
        'neighbourhood_Merton', 'neighbourhood_Newham', 'neighbourhood_Redbridge',
        'neighbourhood_Richmond upon Thames', 'neighbourhood_Southwark', 'neighbourhood_Sutton',
        'neighbourhood_Tower Hamlets', 'neighbourhood_Waltham Forest', 'neighbourhood_Wandsworth',
        'neighbourhood_Westminster',
        'room_type_Entire home/apt', 'room_type_Hotel room',
        'room_type_Private room', 'room_type_Shared room',
        'dist_to_center'
    ]
    input_df = input_df[model_features]

    # Predict price
    prediction_log = model.predict(input_df)[0]       # model outputs log(price)
    prediction = np.expm1(prediction_log)            # reverse log
    st.subheader(f"Predictesadasdd Price: £{prediction:.2f}") #diplay prediction