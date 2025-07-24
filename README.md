#   Crime Hotspot Prediction System

## 📌Overview

This project implements a machine learning system to detect and predict crime hotspots in a city using historical crime data. The system combines geospacial clustering with time-based features to identify current hotspots and forecasts future high risk areas. The system helps law enforcement and city planners to allocate resources more effectively by visualizing dangerous areas on an interactive map.

## 📊 SDGs Alignment
### 🎯 SDG 11: Sustainable Cities and Communities
- This project helps identify high-crime areas to improve urban safety, supports data-driven policing and urban planning and reduces crime risks in public spaces.

### ⚖️ SDG 16: Peace, Justice and Strong Institutions
- This projects helps predict crime patterns to prevent violence, helps law enforcement allocate resources efficiently and promote evidence-based policymaking for safer communities.

## ✨ Features 

- **Data Processing:** Extracts key time-based features (hour, day of the week, month) from the crimestamps.
- **Hotspot Detection:** Uses DBSCAN clustering to identify high-density crime zones.
- **Predictive Model: Random Forest Classifier** forecasts future hotspots with probability scores.
- **Interactive Visualization:** Generates a Folium map showing:
   - 🔴 **Current hotspots** (red markers).
   - 🔵 **Predicted hotspots** (blue markers with probability popups).

## ⚙️ Tools and libraries

- python 3.7+
- pandas
- numpy
- scikit-learn
- geopandas
- folium
