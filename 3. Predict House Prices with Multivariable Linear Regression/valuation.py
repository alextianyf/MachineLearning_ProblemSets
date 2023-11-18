from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np


# Gather Data
california_dataset = fetch_california_housing()
data = pd.DataFrame(data=california_dataset.data, columns=california_dataset.feature_names)
features = data.drop(['Population'], axis=1)

log_prices = np.log(california_dataset.target)
target = pd.DataFrame(log_prices,columns=['PRICE'])

MedInc_IDX = 0
HouseAge_IDX = 1
AveRooms_IDX = 2
AveBedrms_IDX = 3
AveOccup_IDX = 4
Latitude_IDX = 5
Longtitude_IDX = 6
PRICE_IDX = 7

# property_stats = np.ndarray(shape=(1,7))
# property_stats[0][MedInc_IDX] = features['MedInc'].mean()

property_stats = features.mean().values.reshape(1,7)

regr = LinearRegression().fit(features,target)
fitted_vals = regr.predict(features)

MSE = mean_squared_error(target,fitted_vals)
RMSE = np.sqrt(MSE)


# Lets keep this model simple 
def get_log_estimate(income,houseAge,roomNumber, high_confidence=True):
    # Configure property
    property_stats[0][MedInc_IDX] = income
    property_stats[0][HouseAge_IDX] = houseAge
    property_stats[0][AveRooms_IDX] = roomNumber
    
    # Make prediction
    log_estimate = regr.predict(property_stats)[0][0]

    # Calc Range
    if high_confidence:
        upper_bound = log_estimate + 2*RMSE
        lower_bound = log_estimate - 2*RMSE
        interval = 95
    else:
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE
        interval = 68

    return log_estimate, upper_bound, lower_bound, interval

np.median(california_dataset.target)

# We assume California median price today is $1,000,000
# This value can be adjust accordingly
# We will converts the log price estimate using 1970s prices

MEDIAN_PRICE = 10 # unit in 100,000
SCALE_FACTOR = MEDIAN_PRICE/np.median(california_dataset.target)

log_est, upper, lower, conf = get_log_estimate(2, 25, 3)

# Convert to today's dollars
dollar_est = np.e**log_est * 100000 * SCALE_FACTOR
dollar_high = np.e**upper * 100000 * SCALE_FACTOR
dollar_low = np.e**lower * 100000 * SCALE_FACTOR

# Round the dollar values to nearest thousand
rounded_est = np.around(dollar_est, -3)
rounded_high = np.around(dollar_high, -3)
rounded_low = np.around(dollar_low, -3)

print(f'The estimated property value is {rounded_est}.')
print(f'At {conf}% confidence the valuation range is')
print(f'USD {rounded_low} at the lower end to US {rounded_high} at the high end.')

def get_dollar_estimate(income,houseAge,roomNumber, high_confidence=True):
    """
    Estimate the price of a property in California

    Keyword arguments:
    income -- median income in the region
    houseAge -- house age number
    roomNumber -- the number of Room in the property
    high_confidence -- Ture for 95% prediction interval, False for 68% interval
    """

    if roomNumber < 1 or houseAge < 1 or income < 1:
        print('That is unrealistic. Try again.')
        return

    log_est, upper, lower, conf = get_log_estimate(income,houseAge,roomNumber)

    # Convert to today's dollars
    dollar_est = np.e**log_est * 100000 * SCALE_FACTOR
    dollar_high = np.e**upper * 100000 * SCALE_FACTOR
    dollar_low = np.e**lower * 100000 * SCALE_FACTOR

    # Round the dollar values to nearest thousand
    rounded_est = np.around(dollar_est, -3)
    rounded_high = np.around(dollar_high, -3)
    rounded_low = np.around(dollar_low, -3)

    print(f'The estimated property value is ${rounded_est}.')
    print(f'At {conf}% confidence the valuation range is')
    print(f'USD {rounded_low} at the lower end to US {rounded_high} at the high end.')
        