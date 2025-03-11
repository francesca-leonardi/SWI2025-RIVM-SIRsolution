import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize
import pandas as pd
import os
import sys
import datetime
from datetime import datetime


def extract_data(df, city, start, end):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    df_city = df[df['RWZI_AWZI_name'] == city]
    df_sample = df_city[(pd.to_datetime(df_city['Date_measurement'])>= start) & (pd.to_datetime(df_city['Date_measurement'])<=end)]
    df_sample = df_sample.sort_values(by='Date_measurement')
    time_measured = (end - start).days
    measurements = df_sample.iloc[:, -1].to_numpy()
    dates = pd.to_datetime(df_sample['Date_measurement'])
    days = [(date-dates.iloc[0]).days for date in dates]
    return days, measurements, time_measured

file_path = "COVID-19_rioolwaterdata.csv"
df = pd.read_csv(file_path, delimiter=';')

# Select two cities
cities = ["Utrecht", "Amsterdam West"]
# Select a timeframe
date_start, date_end = "2023-12-01", "2024-02-01"
# Select 'True' if you want to display the interval of accepted new values, otherwise select 'False'
error_on = False

time = extract_data(df, cities[0], date_start, date_end)[2]
time_vector = range(time+5)

# Prepare xdata and ydata for both cities
xdata_list1,xdata_list2, ydata_list = [],[],[]
i=0
for city in cities:
    ydata = extract_data(df, city, date_start, date_end)[1]
    xdata = extract_data(df, city, date_start, date_end)[0]
    ydata_list.append(ydata / 1e5)  # Normalize per 100k
    xdata_list1.append(np.array(xdata) + time*i)
    xdata_list2.append(np.array(xdata))
    i=+1

ydata = np.concatenate(ydata_list)
xdata = np.concatenate(xdata_list1)

# SIR Model
def sir_model(y, x, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / (S+I+R)
    dIdt = beta * S * I / (S+I+R) - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Function to fit both cities simultaneously
def fit_odeint_indata(x, beta, gamma, S0_1, I0_1, R0_1, S0_2, I0_2):
    sol1 = integrate.odeint(sir_model, (S0_1, I0_1, R0_1), time_vector, args=(beta, gamma))[:, 1]
    sol2 = integrate.odeint(sir_model, (S0_2, I0_2, (S0_1+I0_1+R0_1-S0_2-I0_2)), time_vector, args=(beta, gamma))[:, 1]
    sol1_indata = np.take(sol1,xdata_list2[0])
    sol2_indata = np.take(sol2,xdata_list2[1])
    return np.concatenate([sol1_indata, sol2_indata])#, np.concatenate([sol1, sol2])


def fit_odeint(x, beta, gamma, S0_1, I0_1, R0_1, S0_2, I0_2):
    sol1 = integrate.odeint(sir_model, (S0_1, I0_1, R0_1), time_vector, args=(beta, gamma))[:, 1]
    sol2 = integrate.odeint(sir_model, (S0_2, I0_2, (S0_1+I0_1+R0_1-S0_2-I0_2)), time_vector, args=(beta, gamma))[:, 1]
    return  [sol1, sol2]



# Fit the model
popt, _ = optimize.curve_fit(fit_odeint_indata, xdata, ydata, bounds=((1e-2,1e-2,1e10,0,0,1e10,0),(2,2,2e12,3e10,1e12,2e12,3e10)))

# Extract optimized parameters
beta_opt, gamma_opt, S0_1, I0_1, R0_1, S0_2, I0_2 = popt

# Solve with optimized parameters
fitted = fit_odeint(xdata, beta_opt, gamma_opt, S0_1, I0_1, R0_1, S0_2, I0_2)

# Error estimate
if error_on:
  error=np.zeros(2)
  error[0]=np.sqrt(np.sum(np.square(fit_odeint_indata(xdata, beta_opt, gamma_opt, S0_1, I0_1, R0_1, S0_2, I0_2)[0:len(ydata_list[0])]-ydata_list[0]))/(len(ydata_list[0]-1)))
  error[1]=np.sqrt(np.sum(np.square(fit_odeint_indata(xdata, beta_opt, gamma_opt, S0_1, I0_1, R0_1, S0_2, I0_2)[len(ydata_list[0]):]-ydata_list[1]))/(len(ydata_list[1]-1)))

# Create subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# City 1 Plot (Utrecht)
axs[0].scatter(xdata_list2[0], ydata_list[0], label=f'City 1 Data ({cities[0]})', color="red")
axs[0].plot(time_vector[0:-1], fitted[0][0:-1], label=f"Fitted SIR ({cities[0]})", color="green")
axs[0].set_xlabel("Days")
axs[0].set_ylabel("Viral load (RNA) per person per day")
axs[0].legend()
axs[0].set_title(f"SIR Model Fitting - {cities[0]} (from {date_start} to {date_end})")

# City 2 Plot (Amsterdam West)
axs[1].scatter(xdata_list2[1], ydata_list[1], label=f'City 2 Data ({cities[1]})', color="blue")
axs[1].plot(time_vector[0:-1], fitted[1][0:-1], label=f"Fitted SIR ({cities[1]})", color="green")
axs[1].set_xlabel("Days")
axs[1].set_ylabel("Viral load (RNA) per person per day")
axs[1].legend()
axs[1].set_title(f"SIR Model Fitting - {cities[1]} (from {date_start} to {date_end})")

# Plot error intervals
if error_on:
    axs[0].fill_between(time_vector[-4:-1], (fitted[0][-4:-1]-error[0]), (fitted[0][-4:-1]+error[0]), color='b', alpha=.1)
    axs[1].fill_between(time_vector[-4:-1], (fitted[1][-4:-1]-error[1]), (fitted[0][-4:-1]+error[1]), color='b', alpha=.1)

# Show plots
plt.tight_layout()
plt.show()

# Print estimated parameters
print(f"Estimated β: {beta_opt:.3f}")
print(f"Estimated γ: {gamma_opt:.3f}")
print(f"Estimated V: {(S0_1+I0_1+R0_1):.0f}")
print(f"City 1 Initial S: {(S0_1/(S0_1+I0_1+R0_1)):.5f}, I: {(I0_1/(S0_1+I0_1+R0_1)):.5f}, R: {R0_1/(S0_1+I0_1+R0_1):.5f}")
print(f"City 2 Initial S: {(S0_2/(S0_1+I0_1+R0_1)):.5f}, I: {(I0_2/(S0_1+I0_1+R0_1)):.5f}, R: {((S0_1+I0_1+R0_1-S0_2-I0_2)/(S0_1+I0_1+R0_1)):.5f}")
print(f"Estimated R0: {beta_opt / gamma_opt:.3f}")
