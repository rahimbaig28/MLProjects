Air Quality Index (AQI) Prediction Project

Overview

This project predicts the Air Quality Index (AQI) by analyzing historical environmental and meteorological data. The dataset includes features like temperature, atmospheric pressure, humidity, visibility, wind speed, and particulate matter (PM 2.5). The project involves data scraping, preprocessing, feature engineering, and machine learning model evaluation to identify the best prediction model.


---

Dataset Description

The dataset includes the following columns:

T: Mean Temperature (°C).

TM: Maximum Temperature (°C).

Tm: Minimum Temperature (°C).

SLP: Sea Level Pressure (hPa).

H: Mean Humidity (%).

VV: Mean Visibility (km).

V: Mean Wind Speed (km/h).

VM: Maximum Wind Speed (km/h).

PM 2.5: Particulate Matter concentration (µg/m³).



---

Workflow

1. Data Collection

Scraped environmental and air quality data from [Website Name/URL] using Python libraries like BeautifulSoup or Selenium.

Collected yearly data to ensure a robust and diverse dataset.


2. Data Preprocessing

Combined multiple years of data into a single dataset.

Cleaned the dataset by:

Handling missing values (e.g., mean imputation for continuous variables).

Removing duplicates.

Converting data types to appropriate formats.



3. Feature Engineering

Derived new features to improve model accuracy, such as:

Temperature range: TM - Tm.

Humidity-pressure interaction terms.

Wind Speed Index: V / VM.

Seasonal features (e.g., Winter, Summer).

Time-based features (e.g., Month, Day).



4. Model Training and Evaluation

Split the data into training and testing sets.

Applied and evaluated the following machine learning models:

Linear Regression

Decision Trees

Random Forest

Gradient Boosting (XGBoost, LightGBM)

Neural Networks


Metrics used for evaluation:

Mean Absolute Error (MAE)

Root Mean Square Error (RMSE)

R-squared (R²)



5. Best Model Selection

Identified the best-performing model based on the evaluation metrics.

Performed hyperparameter tuning to optimize model performance.