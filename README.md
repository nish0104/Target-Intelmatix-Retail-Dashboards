# Target-Intelmatix-Retail-Dashboards

**Contributors:** Nishthaben Vaghani, Hinal Sachpara, Darpan Radadiya, Manideep Mallurwar  
**Institution:** Northeastern University – ALY6080: Integrated Experiential Learning  
**Client:** Intelmatix Corporation  
**Tools & Libraries:** Python, Streamlit, Pandas, Plotly, Geopandas, Scikit-learn, Folium, Branca   
**Duration:** Mar 2025 – Jul 2025  

---

##  Project Overview

This experiential project aims to deliver **interactive dashboards** that help Intelmatix and its enterprise clients make **location-aware business decisions** using **SafeGraph mobility data** and demographic insights.

We developed **four unique dashboards** focused on:

1. **Target Retail Foot Traffic Forecasting**  
2. **Ulta Beauty Customer Segmentation**  
3. **Tesla EV Charging Siting**  
4. **Localized Advertising & Demographic Targeting**

Each dashboard serves a real-world planning function such as staffing, inventory, infrastructure siting, and media buying.

---

##  Data Sources

- **SafeGraph Weekly Patterns** – foot traffic and dwell time
- **U.S. Census via tidycensus API** – demographics, income, race, age
- **Geospatial shapefiles** – block groups and POI locations

---

##  Team Contributions

| Team Member          | Contribution                                                             |
|----------------------|--------------------------------------------------------------------------|
| Nishthaben Vaghani   | Target foot traffic prediction model + dashboard (Linear Regression)     |
| Hinal Sachpara       | Ulta customer clustering using K-Means                                   |
| Darpan Radadiya      | Tesla charger siting using gravity scoring and geospatial mapping        |
| Manideep Mallurwar   | Advertising dashboard with scoring by demographics and dwell time        |

---

##  Tools & Techniques

- **Python & Streamlit**: Dashboard development  
- **Plotly, Folium**: Visualizations and interactive maps  
- **Pandas, Scikit-learn**: Modeling and data transformation  
- **Geopandas, Branca**: Mapping and demographic overlays  
- **Models Used**: Linear Regression, K-Means, Custom Weighted Scoring  

---

##  Dashboards Summary

### 1.  Target Forecasting (by Nishthaben Vaghani)
- Forecast weekly visits using Linear Regression (R² ≈ 0.85)
- Visualize trends, dwell time, and readiness metrics
- Optimize staffing and inventory planning

### 2.  Ulta Customer Segmentation (by Hinal Sachpara)
- Segment stores by visit behavior using K-Means
- Highlight demographic drivers (age, race, income)
- Tailor promotions by cluster

### 3.  Tesla EV Charging Planner (by Darpan Radadiya)
- Score zones based on foot traffic, dwell time, EV adoption
- Use gravity model to prioritize siting decisions
- Map coverage gaps and station proximity

### 4.  Ad Placement Dashboard (by Manideep Mallurwar)
- Score census block groups using visits × dwell time
- Filter zones by dominant age, race, income
- Optimize localized campaigns with ROI potential

---

##  Business Impact

- Enable **data-driven operational decisions** for major retailers and mobility companies
- Improve **resource allocation**, infrastructure planning, and **localized targeting**
- Enhance Intelmatix’s EDIX platform with behavioral intelligence

---


##  References

- [SafeGraph Documentation](https://docs.safegraph.com)
- [tidycensus R Package](https://walker-data.com/tidycensus/)
- Suhara et al. (2021) – Gravity Models in Human Mobility
- [Streamlit Docs](https://docs.streamlit.io/)
- [Plotly Python](https://plotly.com/python/)
