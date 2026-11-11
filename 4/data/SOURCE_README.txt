RusWeather-GF: Gap-Filled Daily Weather Dataset for Russia (1980-2023)
================================================================================

DATASET NAME
------------
RusWeather-GF (Russian Weather - Gap-Filled)

FULL TITLE
----------
Gap-Filled Daily Weather Dataset for Russia (1980-2023): 
Temperature and Precipitation from 593 Stations

DESCRIPTION
-----------
RusWeather-GF provides daily meteorological observations (temperature and 
precipitation) from 593 Russian weather stations spanning 44 years (1980-2023), 
integrated with elevation data from FABDEM.

All missing values in temperature and precipitation have been filled using a 
validated multi-method approach, ensuring 100% data completeness while 
maintaining temporal and spatial consistency.

STATISTICS
----------
Records:              8,893,613
Stations:             593
Period:               1980-01-01 to 2023-12-31 (44 years)
Temporal resolution:  Daily
Spatial coverage:     Russia (19°-180°E, 41°-82°N)

VARIABLES
---------
station_id      - WMO station identifier (string)
date            - Observation date in YYYY-MM-DD format (date)
year            - Year (integer)
month           - Month, 1-12 (integer)
day             - Day of month, 1-31 (integer)
temperature     - Daily mean air temperature in °C (float, gap-filled)
precipitation   - Daily precipitation sum in mm (float, gap-filled)
lat             - Station latitude in decimal degrees (float)
lon             - Station longitude in decimal degrees (float)
elevation       - Station elevation above sea level in meters (float, FABDEM v1.2)

DATA QUALITY AND GAP-FILLING
-----------------------------
Missing values: 100% filled
Original missing data:
  - Temperature: ~1.6% of observations
  - Precipitation: ~1.8% of observations

Gap-filling methods (selected based on gap length):
  1. Short gaps (≤7 days):    Inverse Distance Weighting (IDW) interpolation
                               using 5+ neighboring stations within 400 km
  
  2. Medium gaps (8-30 days): Random Forest regression with temporal, spatial,
                               and topographic features
  
  3. Long gaps (>30 days):    Station-specific climatological means

Quality validation:
  - Temporal consistency: Autocorrelation preserved (Δ < 0.01)
  - Spatial consistency: Correlation r > 0.999 between original and gap-filled data
  - Visual inspection: No artificial discontinuities at gap boundaries

DATA SOURCES
------------
Meteorological data:
  All-Russian Research Institute of Hydrometeorological Information - 
  World Data Centre (RIHMI-WDC)

Elevation data:
  FABDEM v1.2 (Forest And Buildings removed Copernicus Digital Elevation Model)
  https://data.bris.ac.uk/data/dataset/s5hqmjcdj8yo2ibzi9b4ew3sn

USAGE NOTES
-----------
- All timestamps are in UTC
- Temperature represents daily mean (not instantaneous measurement)
- Precipitation is daily cumulative sum (00:00 to 23:59 UTC)
- Gap-filled values are statistically validated but users should consider
  the gap-filling methodology when interpreting results for specific analyses

RECOMMENDED CITATION
--------------------
Tkachenko M.A., Fomin D.S. (2025). RusWeather-GF: Gap-Filled Daily Weather Dataset for 
Russia (1980-2023) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.17789545

LICENSE
-------
This dataset is licensed under Creative Commons Attribution 4.0 International 
(CC BY 4.0). 

You are free to:
  - Share: copy and redistribute the material in any medium or format
  - Adapt: remix, transform, and build upon the material for any purpose
  - Commercial use: yes, allowed

Under the following terms:
  - Attribution: You must give appropriate credit, provide a link to the 
    license, and indicate if changes were made

Full license: https://creativecommons.org/licenses/by/4.0/

CONTACT
-------
Tkachenko Margarita Aleksandrovna
V.V. Dokuchaev Soil Science Institute
Pyzhevsky Lane 7, Moscow, Russia
Email: usa4eva.m@mail.ru

VERSION
-------
Version 1.0 (2025): Initial release