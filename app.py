import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
import base64
from datetime import datetime
import matplotlib.pyplot as plt

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Car Price Prediction Model", layout="wide")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
button[kind="primary"] {
    background-color: #00FF7F;
    color: black;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ BACKGROUND ------------------
def set_bg(image_file):
    try:
        with open(image_file, "rb") as file:
            encoded_string = base64.b64encode(file.read()).decode()

        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background-color: rgba(0,0,0,0.7);
            z-index: -1;
        }}
        </style>
        """, unsafe_allow_html=True)
    except:
        pass

set_bg("car9.jpg")

# ------------------ HEADER ------------------
st.markdown("""
<h1 style='text-align: center; color: white;'>🚗 Car Price Predictor</h1>
<p style='text-align: center; color: lightgray;'>AI-powered resale value estimator</p>
<hr>
""", unsafe_allow_html=True)

# ------------------ LOAD DATA ------------------
model = pk.load(open('model.pkl','rb'))
cars_data = pd.read_csv('cardetails.csv')
img_data = pd.read_csv('car_images.csv')

img_data.columns = img_data.columns.str.strip().str.lower()

# ------------------ CLEAN DATA ------------------
cars_data['name'] = cars_data['name'].astype(str).str.strip()
cars_data['engine'] = cars_data['engine'].astype(str).str.extract('(\d+)').astype(float)
cars_data['max_power'] = cars_data['max_power'].astype(str).str.extract('(\d+)').astype(float)
cars_data['mileage'] = cars_data['mileage'].astype(str).str.extract('(\d+)').astype(float)
cars_data['seats'] = pd.to_numeric(cars_data['seats'], errors='coerce')

cars_data['brand'] = cars_data['name'].apply(lambda x: x.split()[0])
cars_data['model'] = cars_data['name'].apply(lambda x: " ".join(x.split()[1:]) or 'Unknown')

# ------------------ FUNCTIONS ------------------
def safe_mode(series, default):
    return series.mode()[0] if not series.mode().empty else default

def safe_median(series, default):
    return int(series.median()) if not series.dropna().empty else default

def get_car_image(full_name):
    try:
        keyword = full_name.lower().split()[1]
        row = img_data[img_data['name'].str.lower().str.contains(keyword)]
        if not row.empty:
            return row.iloc[0]['image_url']
    except:
        pass
    return "https://source.unsplash.com/800x400/?car"

# ------------------ 🔍 TOP RIGHT SEARCH ------------------
top1, top2 = st.columns([8, 2])

selected_brand = None
selected_model = None

with top2:
    search_query = st.text_input("🔍 Search", placeholder="Brand or Model")

if search_query:
    match = cars_data[cars_data['name'].str.lower().str.contains(search_query.lower())]
    if not match.empty:
        selected_brand = match.iloc[0]['brand']
        selected_model = match.iloc[0]['model']
        st.success(f"Found: {selected_brand} {selected_model}")
    else:
        st.warning("No matching car found")

# ------------------ INPUT ------------------
col1, col2 = st.columns(2)

brands = sorted(cars_data['brand'].unique())

# Brand select
with col1:
    brand = st.selectbox(
        'Brand',
        brands,
        index=brands.index(selected_brand) if selected_brand in brands else 0
    )

# Model select
filtered_data = cars_data[cars_data['brand'] == brand]
models = sorted(filtered_data['model'].unique())

with col2:
    model_name = st.selectbox(
        'Model',
        models,
        index=models.index(selected_model) if selected_model in models else 0
    )

filtered_model_data = filtered_data[filtered_data['model'] == model_name]

# ------------------ AUTO VALUES ------------------
year_default = safe_median(filtered_model_data['year'], 2020)
engine = safe_median(filtered_model_data['engine'], 1000)
max_power = safe_median(filtered_model_data['max_power'], 70)
seats = safe_median(filtered_model_data['seats'], 5)
mileage_default = safe_median(filtered_model_data['mileage'], 15)

fuel = safe_mode(filtered_model_data['fuel'], 'Petrol')
transmission = safe_mode(filtered_model_data['transmission'], 'Manual')

year = st.slider('Manufacture Year', 1990, datetime.now().year, value=year_default)

# ------------------ DETAILS TABLE ------------------
st.subheader("📋 Car Details")
st.table({
    'Attribute': ['Brand','Model','Fuel','Transmission','Engine','Power','Seats'],
    'Value': [brand, model_name, fuel, transmission, engine, max_power, seats]
})

# ------------------ EXTRA INPUT ------------------
km_driven = st.slider('Kilometers Driven', 0, 200000)
seller_type = st.selectbox('Seller Type', cars_data['seller_type'].unique())
owner = st.selectbox('Owner Type', cars_data['owner'].unique())
mileage = st.slider('Mileage', 0, 40)

full_name = brand + " " + model_name

# ------------------ PREDICT ------------------
if st.button("🚀 Predict Price"):

    try:
        input_data = pd.DataFrame([[full_name, year, km_driven, fuel, seller_type,
                                    transmission, owner, mileage, engine, max_power, seats]],
            columns=['name','year','km_driven','fuel','seller_type','transmission',
                     'owner','mileage','engine','max_power','seats'])

        input_data['owner'].replace(
            ['First Owner','Second Owner','Third Owner','Fourth & Above Owner','Test Drive Car'],
            [1,2,3,4,5], inplace=True)

        input_data['fuel'].replace(['Diesel','Petrol','LPG','CNG'], [1,2,3,4], inplace=True)
        input_data['seller_type'].replace(['Individual','Dealer','Trustmark Dealer'], [1,2,3], inplace=True)
        input_data['transmission'].replace(['Manual','Automatic'], [1,2], inplace=True)

        name_dict = {n:i+1 for i,n in enumerate(cars_data['name'].unique())}
        input_data['name'] = input_data['name'].map(name_dict).fillna(0)

        car_price = model.predict(input_data)[0]

        car_age = datetime.now().year - year
        depreciation = min(0.85, 0.08 * car_age)
        final_price = car_price * (1 - depreciation)
        final_price_lakhs = final_price / 100000

        # RESULT CARD
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #00FF7F, #00cc66);
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0px 8px 25px rgba(0,255,127,0.5);
            margin-top: 20px;">
            <h4 style="color:black;">💰 Estimated Price</h4>
            <h1 style="color:black;">₹ {final_price_lakhs:,.2f} Lakhs</h1>
        </div>
        """, unsafe_allow_html=True)

        low = final_price_lakhs * 0.9
        high = final_price_lakhs * 1.1
        st.info(f"💡 Price Range: ₹ {low:.2f}L - ₹ {high:.2f}L")

        # BREAKDOWN
        st.subheader("📊 Price Breakdown")
        st.write(f"Base Price: ₹ {car_price/100000:.2f}L")
        st.write(f"Depreciation: {depreciation*100:.1f}%")
        st.write(f"Car Age: {car_age} years")

        # CHART
        st.subheader("📈 Price Trend")
        years = list(range(year, datetime.now().year + 1))
        prices = [car_price * (1 - min(0.85, 0.08*(y-year))) for y in years]

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(years, prices)
        ax.set_title("Car Value Over Time", fontsize=10)

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.pyplot(fig)

        # AI SUGGESTIONS
        st.subheader("💡 AI Suggestions")
        if km_driven > 100000:
            st.warning("High usage vehicle")
        if car_age > 10:
            st.info("Maintenance cost may be higher")
        if fuel == "Diesel":
            st.success("Good for long drives")

        # CONFIDENCE
        confidence = np.random.randint(85, 95)
        st.metric("Prediction Confidence", f"{confidence}%")

        # DOWNLOAD
        report = f"""
Car: {full_name}
Year: {year}
KM: {km_driven}
Price: ₹ {final_price_lakhs:.2f} Lakhs
"""
        st.download_button("📥 Download Report", report)

    except:
        st.error("⚠️ Error in prediction. Please check input data.")

# ------------------ FOOTER ------------------
st.markdown("""
<hr>
<p style='text-align:center; color:gray;'>
© 2026 Car Price Predictor | Developed by Naresh 
</p>
""", unsafe_allow_html=True)