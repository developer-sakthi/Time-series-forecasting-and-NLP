# Stationarity tests
from statsmodels.tsa.stattools import adfuller, kpss

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
adf_statistic, adf_p_value, adf_critical_values = adfuller(data)
kpss_statistic, kpss_p_value, kpss_critical_values = kpss(data)

# Training the ARIMA model and predicting values
from statsmodels.tsa.arima.model import ARIMA


train = data[:8]
test = data[8:]

model = ARIMA(train, order=(1, 1, 1))
model_fit = model.fit()
predictions = model_fit.predict(test)

# residuals
residuals = model_fit.resid

# Training the SARIMAX model and predicting values
from statsmodels.tsa.statespace.sarimax import SARIMAX


data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
exog = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

model = SARIMAX(data, exog=exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit()
predictions = model_fit.predict()


# AIC and BIC
aic = model_fit.aic
bic = model_fit.bic

# ACF and PACF plots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(residuals)
plot_pacf(residuals)


# Rolling mean and standard deviation
import pandas as pd

data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
rolling_mean = data.rolling(window=3).mean()
rolling_std = data.rolling(window=3).std()


