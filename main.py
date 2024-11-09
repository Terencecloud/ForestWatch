import streamlit as st
import plotly
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine
import re
import os

# Streamlit layout configuration
st.set_page_config(layout="wide")

# Initialize session state
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = False
if 'current_readings' not in st.session_state:
    st.session_state.current_readings = False

# Database connection
DATABASE_URL = 'postgresql://postgres:Smallholder19@localhost/forestwatch'
engine = create_engine(DATABASE_URL)

# Function to fetch data from the database
def fetch_data():
    try:
        query = """
        SELECT
            id,
            image_name,
            deforestation_percentage,
            deforested_area_m2,
            remaining_area_m2,
            remaining_forest_percentage,
            ndvi_highest,
            ndvi_mean,
            ndvi_lowest
        FROM
            deforestation_results
        ORDER BY
            id ASC;
        """
        return pd.read_sql(query, engine)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

# Function to extract year from image name
def extract_year(image_name):
    match = re.search(r'\d{4}', image_name)
    return int(match.group(0)) if match else None

# Function to get the specific image path for recent readings
def get_specific_image_path():
    specific_image_path = 'output_images/2018-10-31-00_00_2018-10-31-23_59_Sentinel-2_L2A_Highlight_Optimized_Natural_Color_.png_deforestation.png'
    return specific_image_path if os.path.exists(specific_image_path) else None

# Streamlit layout
st.markdown("<h1 style='text-align: center;'>FOREST WATCH ANALYTICS</h1>", unsafe_allow_html=True)

# Display Area Description Coordinates
coordinates = {
    "type": "Polygon",
    "coordinates": [[[34.180733, -11.535937], [34.047043, -11.535937],
                     [34.047043, -11.401216], [34.180733, -11.401216],
                     [34.180733, -11.535937]]]
}
coords_df = pd.DataFrame(coordinates["coordinates"][0], columns=["Longitude", "Latitude"])
total_area = 200000  # Placeholder for actual area calculation logic

st.markdown("<h2 style='text-align: center;'>Area Description Coordinates</h2>", unsafe_allow_html=True)
st.table(coords_df)
st.write(f"**Total Area**: {total_area} m²")

aoimap_image_path = 'data/AoImap.jpeg'
if os.path.exists(aoimap_image_path):
    st.image(aoimap_image_path, caption="Area of Interest Map", use_column_width=True)
else:
    st.warning("Area of Interest Map image not found.")

# CSS for green buttons
st.markdown("""
    <style>
    .stButton>button {
        background-color: green;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Layout for buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("Current Readings"):
        st.session_state.current_readings = True
        st.session_state.historical_data = False  # Reset historical data state

with col2:
    if st.button("Historical Data"):
        st.session_state.historical_data = True
        st.session_state.current_readings = False  # Reset current readings state

# Fetch data from the database
df = fetch_data()

# Display Current Readings
if st.session_state.get('current_readings'):
    st.title("Current Readings")
    if not df.empty:
        latest_reading = df.iloc[-1]
        image_name = latest_reading['image_name']
        
        # Display detailed information
        st.write(f"**Image Name**: {image_name}")
        date = image_name.split('-')[0] + '-' + image_name.split('-')[1] + '-' + image_name.split('-')[2][:2]
        satellite = "_".join(image_name.split('_')[4:6])
        image_band = image_name.split('_')[-1].split('.')[0]
        st.write(f"**Date**: {date}")
        st.write(f"**Satellite**: {satellite}")
        st.write(f"**Image Band**: {image_band}")
        st.write(f"**Deforestation Percentage**: {latest_reading['deforestation_percentage']:.2f}%")
        st.write(f"**Deforested Area**: {latest_reading['deforested_area_m2']:,} m²")
        st.write(f"**Remaining Forest Area**: {latest_reading['remaining_area_m2']:,} m²")
        st.write(f"**Remaining Forest Percentage**: {latest_reading['remaining_forest_percentage']:.2f}%")
        st.write(f"**NDVI Highest**: {latest_reading['ndvi_highest']:.4f}")
        st.write(f"**NDVI Mean**: {latest_reading['ndvi_mean']:.4f}")
        st.write(f"**NDVI Lowest**: {latest_reading['ndvi_lowest']:.4f}")

        # Display specific image if available
        specific_image_path = get_specific_image_path()
        if specific_image_path:
            st.image(specific_image_path, caption="Most Recent Masked Image", use_column_width=True)
        else:
            st.warning("No masked images available.")
    else:
        st.warning("No current readings available.")

# Display Historical Data
elif st.session_state.get('historical_data'):
    st.title("Historical Data")

    if df is not None and not df.empty:
        st.dataframe(df)  # Display the DataFrame

        # Extract years from image names and add to DataFrame
        df['year'] = df['image_name'].apply(lambda x: x.split('_')[0][:4])

        # Group by year and calculate averages
        yearly_averages = df.groupby('year').agg({
            'deforestation_percentage': 'mean',
            'deforested_area_m2': 'mean',
            'remaining_area_m2': 'mean',
            'ndvi_mean': 'mean'
        }).reset_index()

        # Create 2x2 subplots for historical trends
        fig = make_subplots(rows=2, cols=2, subplot_titles=[
            'Deforestation Percentage (%)',
            'Deforested Area (m²)',
            'Remaining Area (m²)',
            'NDVI Mean'
        ])

        # Add plots
        fig.add_trace(go.Scatter(x=yearly_averages['year'], y=yearly_averages['deforestation_percentage'],
                                 mode='lines+markers', name='Deforestation Percentage (%)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=yearly_averages['year'], y=yearly_averages['deforested_area_m2'],
                                 mode='lines+markers', name='Deforested Area (m²)'), row=1, col=2)
        fig.add_trace(go.Scatter(x=yearly_averages['year'], y=yearly_averages['remaining_area_m2'],
                                 mode='lines+markers', name='Remaining Area (m²)'), row=2, col=1)
        fig.add_trace(go.Scatter(x=yearly_averages['year'], y=yearly_averages['ndvi_mean'],
                                 mode='lines+markers', name='NDVI Mean'), row=2, col=2)

        # Update layout
        fig.update_layout(
            title_text="Historical Trends",
            height=1200,  # Increased height for clarity
            width=1400,
            showlegend=True
        )
        st.plotly_chart(fig)

    else:
        st.error("No historical data available in the database.")
