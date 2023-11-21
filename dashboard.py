import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
from babel.numbers import format_currency
sns.set(style='dark')

# Function untuk byday_df
def create_byday_df(day_df):
    byday_df = day_df.groupby(by="weathersit").instant.nunique().reset_index()
    byday_df.rename(columns={
        "instant": "wths_count"
    }, inplace=True)
    
    return byday_df

# Function untuk byhour_df
def create_byhour_df(hour_df):
    byhour_df = hour_df.groupby(by="weathersit").instant.nunique().reset_index()
    byhour_df.rename(columns={
        "instant": "wths_count"
    }, inplace=True)
    
    return byhour_df

# Function untuk rent_df
def create_rent_df(day_df):
    rent_df = day_df.loc[day_df['yr'] == 1].groupby(by=["mnth"]).agg({
        "casual": "sum",
        "registered": "sum",
        "cnt": "sum",
    })
    return rent_df

# Function untuk result_df sebagai analisa regresi
def create_result_df(day_df):
    features = ['temp', 'atemp', 'hum', 'windspeed', 'holiday', 'weathersit']
    X = day_df[features]
    y = day_df['cnt']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Residuals': y_test - y_pred})

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return result_df, model, mse, r2

# Baca berkas data
day_df = pd.read_csv("data/day.csv")
hour_df = pd.read_csv("data/hour.csv")

# Untuk membantu filter tanggal
datetime_columns = ["dteday"]
day_df.sort_values(by="dteday", inplace=True)
day_df.reset_index(inplace=True)
 
for column in datetime_columns:
    day_df[column] = pd.to_datetime(day_df[column])

# Membuat komponen filter
min_date = day_df["dteday"].min()
max_date = day_df["dteday"].max()

# Filter harian
with st.sidebar:
    # Mengambil start_date & end_date dari date_input
    start_date, end_date = st.date_input(
        label='Time Range For Daily Filters',min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

main_df = day_df[(day_df["dteday"] >= str(start_date)) & 
                (day_df["dteday"] <= str(end_date))]
# Filter Jam
with st.sidebar:
    # Mengambil start_date & end_date dari date_input
    start_date, end_date = st.date_input(
        label='Time Range For Hourly Filters',min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

second_df = hour_df[(hour_df["dteday"] >= str(start_date)) & 
                (hour_df["dteday"] <= str(end_date))]

# Memanggil helper function
byday_df = create_byday_df(main_df)
byhour_df = create_byhour_df(second_df)
rent_df = create_rent_df(main_df)
result_df, model, mse, r2 = create_result_df(main_df)

# === Start make page === 

# Web - Title 

st.header('Bike Sharing Collection Dashboard :bike::sparkles:')

# Web - Favorite Weathersit

st.subheader("Favorite Weathersit\nWeathersit Code Information: \n- 1: Clear, Few clouds, Partly cloudy, Partly cloudy\n- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist\n- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds\n- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog")
st.text('Note: Use daily filters to see daily weathersit counts and \n      hourly filters to see hourly weathersit counts on sidebar')

# Day Weathersit Count

wt_day_count = day_df.weathersit.sum()
st.metric("Total Weathersit Count Each Day", value=wt_day_count)
fig = plt.figure(figsize=(10, 5))
colors = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
sns.barplot(
    y="wths_count", 
    x="weathersit",
    data=byday_df.sort_values(by="wths_count", ascending=False),
    palette=colors
)
plt.title("Number of Weathersit by Day", loc="center", fontsize=15)
plt.ylabel('Weathersit(count)')
plt.xlabel('Weathersit Code')
plt.tick_params(axis='x', labelsize=12)
st.pyplot(fig)

# Hour Weathersit Count

wt_hour_count = hour_df.weathersit.sum()
st.metric("Total Weathersit Count Each Hour", value=wt_hour_count)
fig = plt.figure(figsize=(10, 5))
colors = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
sns.barplot(
    y="wths_count", 
    x="weathersit",
    data=byhour_df.sort_values(by="wths_count", ascending=False),
    palette=colors
)
plt.title("Number of Weathersit by Hour", loc="center", fontsize=15)
plt.ylabel('Weathersit(count)')
plt.xlabel('Weathersit Code')
plt.tick_params(axis='x', labelsize=12)
st.pyplot(fig)

# Web - Rent Perform in 2012

st.subheader("Rent Perform in 2012\nLabel Description Information:\n- Casual: count of casual users \n- Registered: count of registered users \n- CnT: count of total rental bikes including both casual and registered")

fig = plt.figure(figsize=(10, 5))
plt.plot(
    rent_df["casual"],  
    marker='o', 
    linewidth=2,
    )
plt.plot(
    rent_df["registered"],
    marker='o', 
    linewidth=2,
    )
plt.plot(
    rent_df["cnt"],
    marker='o', 
    linewidth=2,
    )
plt.grid(True)
x = np.arange(0, 13, 1)
plt.xticks(x)
plt.legend(rent_df, loc=2)
plt.ylabel('Users(count)')
plt.xlabel('Month')
plt.title("Number of Users per Month (2012)", loc="center", fontsize=20)
st.pyplot(fig)

# Web - Regression Analysis

st.subheader("Regression Analysis\nThis analysis shows a correlation between temp, atemp, hum, windspeed, holiday, and weathersit to bike rent users")

# Display Regression Coefficients
st.write("#### Regression Coefficients:")
st.write("Intercept:", model.intercept_)
st.write("Coefficients:", model.coef_)

# Display Regression Metrics
st.write("#### Regression Metrics:")
st.write("Mean Squared Error:", mse)
st.write("R-squared:", r2)

# Display Residuals Plot
st.write("#### Residuals Plot:")
fig_resid = plt.figure(figsize=(10, 5))
sns.scatterplot(x=result_df['Predicted'], y=result_df['Residuals'])
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residuals Plot")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
st.pyplot(fig_resid)

# Visualisasi hasil prediksi pada Streamlit
st.write("#### Actual vs. Predicted Count of Bike Rentals:")
fig_pred = plt.figure(figsize=(10, 5))
plt.scatter(result_df['Actual'], result_df['Predicted'])
plt.xlabel('Actual Count')
plt.ylabel('Predicted Count')
plt.title('Actual vs. Predicted Count of Bike Rentals')
st.pyplot(fig_pred)
st.subheader('Explanation of Diagram Above: :man-running:\nIf you get a positive regression coefficient and a positive correlation between a particular independent variable (temperature) and the dependent variable (number of bike rentals). A positive value indicates that an increase in that independent variable is positively correlated with an increase in the dependent variable. In this context, if the temperature increases, the number of bike rentals is likely to increase.')