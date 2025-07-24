import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import geopandas as gpd
import folium
from datetime import datetime

# Load and Preprocess the data

crime_data = pd.read_csv('city_crimes_large.csv')

crime_data['datetime'] = pd.to_datetime(crime_data['date'])
crime_data['hour'] = crime_data['datetime'].dt.hour
crime_data['day_of_week'] = crime_data['datetime'].dt.dayofweek
crime_data['month'] = crime_data['datetime'].dt.month

# Convert to GeoDataFrame

gdf = gpd.GeoDataFrame(
    crime_data,
    geometry = gpd.points_from_xy(crime_data.longitude, crime_data.latitude)
)

# Hotspots Detection Using DBSCAN

coords = crime_data[['latitude', 'longitude']].values
db = DBSCAN(eps = 0.002, min_samples= 10).fit(coords)
crime_data['hotspot'] = (db.labels_ !=-1).astype(int)

# Train predictive model

features = ['latitude', 'longitude', 'hour', 'day_of_week', 'month']

X_train, X_test, y_train, y_test = train_test_split(
    crime_data[features],
    crime_data['hotspot'],
    test_size = 0.2
)

model = RandomForestClassifier(n_estimators = 100)
model.fit(X_train, y_train)

print(f"Model Accuracy: {model.score(X_test, y_test):.2f}")

# Generate future hotspot predictions

future_dates  = pd.date_range(start=datetime.today(), periods = 7)

prediction_data = []
for date in future_dates:
    for hour in range(24):
        prediction_data.append({
            'latitude':coords[:,0].mean(),
            'longitude':coords[:,1].mean(),
            'hour': hour,
            'day_of_week': date.dayofweek,
            'month': date.month
        }) 

future_df = pd.DataFrame(prediction_data)
future_df['hotspot_prob'] = model.predict_proba(future_df[features])[:,1]

# Visualize on Interactive map

m = folium.Map(location = [coords[:,0].mean(), coords[:,1].mean()], zoom_start = 13)

# Plot current hotspots(red)

hotspots = crime_data[crime_data['hotspot'] == 1]
for idx, row in hotspots.iterrows():
    folium.CircleMarker(
        location = [row['latitude'], row['longitude']],
        radius = 5,
        color = 'red',
        fill = True
    ).add_to(m)

# plot the predicted hotspots

future_hotspots = future_df[future_df['hotspot_prob'] > 0.7]
for idx, row in future_hotspots.iterrows():
    folium.CircleMarker(
        location = [row['latitude'], row['longitude']],
        radius = 5,
        color = 'blue',
        fill = True,
        popup = f"Time: {row['hour']}:00, Prob: {row['hotspot_prob']:.2f}"
    ).add_to(m)

# Save the map

m.save('crime_hotspots.html')