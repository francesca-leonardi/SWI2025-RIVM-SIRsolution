import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize
import pandas as pd
import os
import sys
import datetime
from datetime import datetime
from scipy.optimize import minimize

#
def extract_data(df, city, start, end):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    df_city = df[df['RWZI_AWZI_name'] == city]
    df_sample = df_city[(pd.to_datetime(df_city['Date_measurement'])>= start) & (pd.to_datetime(df_city['Date_measurement'])<=end)]
    df_sample = df_sample.sort_values(by='Date_measurement')

    # Calculate the difference in days
    time_measured = (end - start).days

    measurements = df_sample.iloc[:, -1].to_numpy()
    dates = pd.to_datetime(df_sample['Date_measurement'])
    days = [(date-dates.iloc[0]).days for date in dates]

    return days, measurements, time_measured


file_path = "COVID-19_rioolwaterdata.csv"
df = pd.read_csv(file_path, delimiter=';')

# Select city
city = "Utrecht"
# Select timeframe
date_start, date_end = "2023-10-01", "2024-02-01"
# Select 'True' if you want to display the interval of accepted new values, otherwise select 'False'
error_on = False

time = extract_data(df, city, date_start, date_end)[2]
time_vector = range(time+5)

# Prepare xdata and ydata
xdata_list, ydata_list = [],[]
ydata = extract_data(df, city, date_start, date_end)[1]
xdata = extract_data(df, city, date_start, date_end)[0]
ydata_list.append(ydata / 1e5)  # Normalize per 100k
xdata_list.append(np.array(xdata))

ydata = np.concatenate(ydata_list)
xdata = np.concatenate(xdata_list)

# SIR Model
def sir_model(y, x, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / (S+I+R)
    dIdt = beta * S * I / (S+I+R) - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Function to fit the solution of the SIR model for one city
def fit_odeint_indata(x, beta, gamma, S0, I0, R0):
    sol = integrate.odeint(sir_model, (S0, I0, R0), time_vector, args=(beta, gamma))[:, 1]
    sol_indata = np.take(sol,xdata_list)
    return np.concatenate(sol_indata)

def fit_odeint(x, beta, gamma, S0, I0, R0):
    sol = integrate.odeint(sir_model, (S0, I0, R0), time_vector, args=(beta, gamma))[:, 1]
    return sol

# Fit the model
popt, _ = optimize.curve_fit(fit_odeint_indata, xdata, ydata, bounds=((1e-2,1e-2,1e10,0,0),(2,2,2e12,3e10,1e12)))

# Extract optimized parameters
beta_opt, gamma_opt, S0, I0, R0 = popt

# Solve with optimized parameters
fitted = fit_odeint(xdata, beta_opt, gamma_opt, S0, I0, R0)

#Error estimate
if error_on:
  error = np.sqrt(np.sum(np.square(fit_odeint_indata(xdata, beta_opt, gamma_opt, S0, I0, R0)[0:len(ydata_list[0])]-ydata_list[0]))/(len(ydata_list[0])-1))

# Create plot
plt.figure(figsize=(10, 5))
plt.scatter(xdata_list, ydata_list, label=f'City Data', color="red")
plt.plot(time_vector[0:-1], fitted[0:-1], label=f"Fitted SIR", color="green")
if error_on:
  plt.fill_between(time_vector[-4:-1], (fitted[-4:-1]-error), (fitted[-4:-1]+error), color='b', alpha=.1)
plt.xlabel("Days")
plt.ylabel("Viral load (RNA) per person per day")
plt.legend()
plt.title(f"SIR Model Fitting - {city} (from {date_start} to {date_end})")

# Show plots
plt.tight_layout()
plt.show()

# Print estimated parameters
print(f"Estimated β: {beta_opt:.3f}")
print(f"Estimated γ: {gamma_opt:.3f}")
print(f"Estimated V: {S0+I0+R0:.0f}")
print(f"Initial S: {S0/(S0+I0+R0):.7f}, I: {I0/(S0+I0+R0):.7f}, R: {R0/(S0+I0+R0):.7f}")
print(f"Estimated r_0: {beta_opt / gamma_opt:.3f}")
