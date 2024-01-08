import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import yfinance as yf 
from scipy.stats import norm

st.title('Stock Price Prediction based on Log Returns')
st.write("Instruction: Please select input ticker and date in the sidebar for computation")

# Create three columns in the Streamlit app
col1, col2, col3 = st.columns(3)

# Input field to enter a stock ticker
with col1:
    ticker = st.text_input('Ticker', placeholder='Eg. AAPL', help="Enter a stock ticker (i.e. AAPL)")

# Date input field to select a start date
with col2:
    start_date = st.date_input('Start Date', help="Select a start date")

# Date input field to select an end date
with col3:
    end_date = st.date_input('End Date', help="Select an end date")

df = yf.download(ticker,start=start_date, end=end_date)
df.reset_index()

st.write(df.head())

st.write('Data Shape')
st.write(df.shape)

st.write('Data Header')
st.write(df.head())

st.write('Checking null values')
st.write(df.isnull().sum())

st.subheader('Log Return Calculation')

st.write('Log Daily returns are calculated by (Final value-Initial Value)/Initial value hence:')
st.write('if the closing price on day 1 is 120, day 2 is 110 and day 3 is 135 then day 2 return would be 22.72% and overall return would be 12.5%')
st.write('However using Log returns = In(final value) - In(initial value) Day 2 return = -0.038 Day 3 = 0.089 Overall = 0.051 Unlike in arthimatic calculation, we get the same return in overall as adding the returns of day 2 and day 3 in log calculation.')

st.write('Log Return')
df['LogReturn'] = np.log(df['Close'])-np.log(df['Close']).shift(1)
st.write(df['LogReturn'])

mu = df['LogReturn'].mean()
sigma = df['LogReturn'].std(ddof=1)

density = pd.DataFrame()
density['x'] = np.arange(df['LogReturn'].min(), df['LogReturn'].max(), 0.001)
density['pdf'] = norm.pdf(density['x'], mu, sigma)
sns.histplot(df['LogReturn'], kde = True, stat = 'density')
plt.plot(density['x'], density['pdf'], color='red')
fig = plt.show()

st.pyplot(fig)

st.write('Blue curve represents the shape of data distribution an the red curve represents Probability distribution function (PDF). Data seems normal hence on the basis of normality assumption, we use CDF  to calculate the increment or decrement of stock price.')

#probability of 5% drop in stock price in one day
st.subheader('Probability of % drop in stock price in one day')
st.write('Probability of 5% drop in stock price in one day')
prob_drop5_day = norm.cdf(-0.05, loc=mu, scale=sigma)
value1 = print('the probability of 5% drop in stock price in one day'+ str(prob_drop5_day))
st.write(prob_drop5_day)

#probability of 10% drop in stock price in one day
st.write('Probability of 10% drop in stock price in one day')
prob_drop10_day = norm.cdf(-0.10, loc=mu, scale=sigma)
print('the probability of 10% drop in stock price in one day'+ str(prob_drop10_day))
st.write(prob_drop10_day)

#probability of 50% drop in stock price in one day
st.write('Probability of 50% drop in stock price in one day')
prob_drop50_day = norm.cdf(-0.5, loc=mu, scale=sigma)
value1 = print('the probability of 50% drop in stock price in one day'+ str(prob_drop50_day))
st.write(prob_drop50_day)

#for yearly stock price drops
mu_yearly = mu * 250
sigma_yearly = sigma * np.sqrt(250)
#for 50% drop
st.subheader('Probability of % drop in stock price in one year')
st.write('Probability of 5% drop in stock price in one day')
prob_drop5_year = norm.cdf(-0.05, mu_yearly, sigma_yearly)
print('the probability of 5% drop in stock price in one year'+ str(prob_drop5_year))
st.write(prob_drop5_year)

st.write('Probability of 25% drop in stock price in one day')
prob_drop25_year = norm.cdf(-0.25, mu_yearly, sigma_yearly)
print('the probability of 25% drop in stock price in one year'+ str(prob_drop25_year))
st.write(prob_drop25_year)



st.write('Probability of 50% drop in stock price in one day')
prob_drop50_year = norm.cdf(-0.50, mu_yearly, sigma_yearly)
print('the probability of 50% drop in stock price in one year'+ str(prob_drop50_year))
st.write(prob_drop50_year)



