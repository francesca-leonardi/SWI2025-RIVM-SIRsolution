Overview

This repository contains two Python scripts that model the spread of COVID-19 using the SIR (Susceptible-Infected-Recovered) epidemiological model.
The model fits wastewater viral load data to estimate key epidemiological parameters such as transmission rate (β), recovery rate (γ), and the basic reproduction number (R0).

Features

- Extracts COVID-19 wastewater data from a CSV file.
- Fits an SIR model to the data using nonlinear least squares optimization.
- Estimates key epidemiological parameters.
- Visualizes model fits and data comparisons.
- Supports fitting data for one or two cities simultaneously.

Dependencies

Ensure you have the following Python libraries installed:
pip install numpy pandas matplotlib scipy

File Descriptions

single_city_SIR.py

Extracts data for a single city.
Fits the SIR model to the extracted data.
Estimates the epidemiological parameters and plots the results.
Displays the fitted model alongside real viral load data.

dual_city_SIR.py

Extracts data for two cities.
Fits the SIR model simultaneously for both cities.
Estimates epidemiological parameters separately for each city.
Plots individual results for both cities.

Usage

For running the Single City Model python single_city_SIR.py
Modify the city, date_start and date_end variables to select a city and timeframe.
Set error_on on True if you want to disply the confidence interval.

For running the Dual City Model python dual_city_SIR.py
Modify the cities list to select two cities for comparison.
Set date_start and date_end for the analysis timeframe.
Set error_on on True if you want to display the confidence interval.

Data Requirements

The script expects a CSV file named COVID-19_rioolwaterdata.csv with at least the following columns:
RWZI_AWZI_name: City name
Date_measurement: Date of measurement
[Last column]: Viral load data
Ensure the CSV file is formatted correctly and stored in the same directory as the scripts.

Outputs

Estimated epidemiological parameters:
Transmission rate (β)
Recovery rate (γ)
Initial conditions (S0, I0, R0)
Basic reproduction number (R0)

Plots visualizing the SIR model fit compared to actual data.



Contact

For questions or contributions, feel free to submit an issue or pull request.
