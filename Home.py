import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from prophet import Prophet
from datetime import datetime, timedelta
import xgboost as xgb

# Set page config as the first Streamlit command
st.set_page_config(page_title="Traffic Analytics Dashboard", layout="wide")

# Define numeric_cols globally, matching the exact column names in the dataset
numeric_cols = ['UNKNOWN', 'PASSENGER CAR', '2 AXLES', '3 AXLES', '4 AXLES', '5 AXLES', '6 AXLES', '7 AXLES', '8 AXLES',
                'BUSES 2 AXLES', 'BUSES 3 AXLES', 'TOTAL BUSES', 'AXLE VEHICLE WEIGHT OVERLOADS',
                'GROSS VEHICLE WEIGHT OVERLOAD >0 <2000', 'GROSS VEHICLE WEIGHT OVERLOAD >=2000',
                'NO GROSS VEHICLE WEIGHT OVERLOADS', 'GROSS VEHICLE WEIGHT COMPLIANCE(%)', 'TOTAL TRAFFIC', 
                'TOTAL TRUCKS', 'TOTAL_OVERLOADS']

# Define a custom Plotly layout for visibility
def apply_custom_plotly_layout(fig):
    fig.update_layout(
        plot_bgcolor='rgba(255, 255, 255, 0.95)',  # White background for the plot area
        paper_bgcolor='rgba(255, 255, 255, 0.95)',  # White background for the entire chart
        font=dict(color='#1e3c72'),  # Dark blue text for contrast
        title_font=dict(color='#1e3c72'),
        xaxis=dict(
            gridcolor='#d3d3d3',  # Light gray gridlines
            tickfont=dict(color='#1e3c72'),
            title_font=dict(color='#1e3c72')
        ),
        yaxis=dict(
            gridcolor='#d3d3d3',
            tickfont=dict(color='#1e3c72'),
            title_font=dict(color='#1e3c72')
        ),
        legend=dict(
            font=dict(color='#1e3c72'),
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='#d3d3d3',
            borderwidth=1
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        hoverlabel=dict(
            font=dict(color='#1e3c72'),
            bgcolor='rgba(255, 255, 255, 0.95)'
        )
    )
    # Ensure lines and markers are visible
    fig.update_traces(
        line=dict(color='#2a5298'),  # Darker blue for lines
        marker=dict(color='#2a5298')  # Darker blue for markers
    )
    return fig

# Cache data loading
@st.cache_data
def load_data(file_path):
    df = pd.read_excel(file_path, sheet_name='GENERAL')
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df = df.sort_values('DATE')
    df['TOTAL_OVERLOADS'] = df['GROSS VEHICLE WEIGHT OVERLOAD >0 <2000'] + df['GROSS VEHICLE WEIGHT OVERLOAD >=2000']
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Debug: Check for missing values before imputation
    missing_counts = df[numeric_cols].isna().sum()
    st.write("Missing values before imputation:", missing_counts[missing_counts > 0].to_dict())
    
    # Fill missing values with median
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    for col in ['TOTAL TRAFFIC', 'TOTAL_OVERLOADS']:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        df[col] = df[col].clip(upper=upper_bound)
    
    df['DAY_OF_WEEK'] = df['DATE'].dt.dayofweek
    return df

# Load data
file_path = r'C:\Users\HomePC\Desktop\data science\GEN SUMMARY ANALYSIS\updated.xlsx'
df = load_data(file_path)

# Streamlit Dashboard
st.markdown("<h1 style='text-align: center; color: #ffffff; text-shadow: 2px 2px 4px #000000;'>Traffic Analytics Dashboard</h1>", unsafe_allow_html=True)

# Dynamic Filters
st.markdown("### Filters")
col1, col2, col3, col4 = st.columns([1, 1, 2, 1])
with col1:
    start_date = st.date_input("Start Date", value=df['DATE'].min(), min_value=df['DATE'].min(), max_value=df['DATE'].max())
with col2:
    end_date = st.date_input("End Date", value=df['DATE'].max(), min_value=df['DATE'].min(), max_value=df['DATE'].max())
with col3:
    selected_stations = st.multiselect("Select Stations", options=['All'] + sorted(df['STATION'].unique().tolist()), default=['All'])
with col4:
    vehicle_categories = ['All', 'Unknowns', 'Passenger Cars', 'Total Trucks', 'Total Buses']
    selected_vehicle_category = st.selectbox("Select Vehicle Category", options=vehicle_categories)

# Apply filters dynamically
filtered_df = df[(df['DATE'].dt.date >= start_date) & (df['DATE'].dt.date <= end_date)]
if 'All' not in selected_stations:
    filtered_df = filtered_df[filtered_df['STATION'].isin(selected_stations)]
filtered_df = filtered_df.reset_index(drop=True)
if 'S/NO' not in filtered_df.columns:
    filtered_df.insert(0, 'S/NO', range(1, len(filtered_df) + 1))
else:
    filtered_df['S/NO'] = range(1, len(filtered_df) + 1)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Tables", "Descriptive", "Predictive", "Prescriptive", "Diagnostic"])

# Tab 1: Tables
with tab1:
    st.markdown("### Data Tables")
    with st.expander("General Summary"):
        totals = filtered_df[numeric_cols].sum(numeric_only=True).to_frame().T
        totals['S/NO'] = 'Total'
        totals['STATION'] = ''
        totals['DATE'] = ''
        summary_df = pd.concat([filtered_df, totals], ignore_index=True)
        st.dataframe(summary_df)
        st.download_button("Download Summary", summary_df.to_csv(index=False), "summary.csv", "text/csv")
    
    with st.expander("Descriptive Statistics (Numerical Columns)"):
        desc_stats = filtered_df[numeric_cols].describe().T
        st.dataframe(desc_stats)
    
    with st.expander("Stations with Missing Data"):
        all_stations = df['STATION'].unique().tolist()
        present_stations = filtered_df['STATION'].unique().tolist()
        missing_stations = [station for station in all_stations if station not in present_stations]
        missing_df = pd.DataFrame({'STATION': missing_stations})
        st.dataframe(missing_df)

# Tab 2: Descriptive Analytics
with tab2:
    st.markdown("### Descriptive Analytics")
    desc_df = filtered_df.copy()
    if selected_vehicle_category != 'All':
        if selected_vehicle_category == 'Unknowns':
            desc_df['TOTAL TRAFFIC'] = desc_df['UNKNOWN']
        elif selected_vehicle_category == 'Passenger Cars':
            desc_df['TOTAL TRAFFIC'] = desc_df['PASSENGER CAR']
        elif selected_vehicle_category == 'Total Trucks':
            desc_df['TOTAL TRAFFIC'] = desc_df['TOTAL TRUCKS']
        elif selected_vehicle_category == 'Total Buses':
            desc_df['TOTAL TRAFFIC'] = desc_df['TOTAL BUSES']
    
    # KPIs
    st.markdown("#### Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Vehicles Weighed", f"{desc_df['TOTAL TRAFFIC'].sum():,.0f}")
    with col2:
        st.metric("Total Overloads", f"{desc_df['TOTAL_OVERLOADS'].sum():,.0f}")
    with col3:
        st.metric("Total Unknowns", f"{desc_df['UNKNOWN'].sum():,.0f}")
    with col4:
        st.metric("Avg Compliance Rate", f"{desc_df['GROSS VEHICLE WEIGHT COMPLIANCE(%)'].mean():.2f}%")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Axle Overloads", f"{desc_df['AXLE VEHICLE WEIGHT OVERLOADS'].sum():,.0f}")
    with col6:
        st.metric("Total Trucks", f"{desc_df['TOTAL TRUCKS'].sum():,.0f}")
    with col7:
        st.metric("Overload >0 <2000", f"{desc_df['GROSS VEHICLE WEIGHT OVERLOAD >0 <2000'].sum():,.0f}")
    with col8:
        st.metric("Overload >=2000", f"{desc_df['GROSS VEHICLE WEIGHT OVERLOAD >=2000'].sum():,.0f}")

    # Visualizations
    st.markdown("#### Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        traffic_per_station = desc_df.groupby('STATION')['TOTAL TRAFFIC'].sum().reset_index()
        fig_traffic = px.bar(traffic_per_station, x='STATION', y='TOTAL TRAFFIC', title="Total Traffic per Station")
        fig_traffic = apply_custom_plotly_layout(fig_traffic)
        st.plotly_chart(fig_traffic, use_container_width=True)
    
    with col2:
        vehicle_dist = pd.DataFrame({
            'Category': ['Unknowns', 'Passenger Cars', 'Total Trucks', 'Total Buses'],
            'Count': [desc_df['UNKNOWN'].sum(), desc_df['PASSENGER CAR'].sum(), desc_df['TOTAL TRUCKS'].sum(), desc_df['TOTAL BUSES'].sum()]
        })
        fig_pie = px.pie(vehicle_dist, names='Category', values='Count', title="Vehicle Distribution")
        fig_pie = apply_custom_plotly_layout(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        compliance_trend = df.groupby('DATE')['GROSS VEHICLE WEIGHT COMPLIANCE(%)'].mean().reset_index()
        fig_compliance = px.line(compliance_trend, x='DATE', y='GROSS VEHICLE WEIGHT COMPLIANCE(%)', title="Compliance Rate Over Time")
        fig_compliance = apply_custom_plotly_layout(fig_compliance)
        st.plotly_chart(fig_compliance, use_container_width=True)
    
    with col4:
        overloads = pd.DataFrame({
            'Type': ['Overload >0 <2000', 'Overload >=2000'],
            'Count': [desc_df['GROSS VEHICLE WEIGHT OVERLOAD >0 <2000'].sum(), desc_df['GROSS VEHICLE WEIGHT OVERLOAD >=2000'].sum()]
        })
        fig_overloads = px.bar(overloads, x='Type', y='Count', title="Total Overloads")
        fig_overloads = apply_custom_plotly_layout(fig_overloads)
        st.plotly_chart(fig_overloads, use_container_width=True)

# Tab 3: Predictive Analytics
with tab3:
    st.markdown("### Predictive Analytics")
    selected_station = st.selectbox("Select Station for Forecasting", options=sorted(df['STATION'].unique().tolist()))
    model_choice = st.selectbox("Choose Forecasting Model", 
                                ["LSTM", "Prophet", "Linear Regression", "XGBoost", "Prophet with Regressors"])
    forecast_steps = st.slider("Forecast Days", 1, 90, 30, key="predictive_slider")
    
    station_data = df[df['STATION'] == selected_station].sort_values('DATE')
    
    @st.cache_resource
    def forecast_data(_data, target_col, model_type, forecast_steps=30):
        data = _data.dropna(subset=[target_col])
        series = data[target_col].values
        if len(series) < 10:
            st.warning(f"Not enough data for {target_col} at {selected_station}. Using historical average.")
            avg = series.mean()
            forecast_dates = pd.date_range(start=data['DATE'].max() + timedelta(days=1), periods=forecast_steps)
            return pd.DataFrame({'DATE': forecast_dates, 'PREDICTED': [avg] * forecast_steps}), None, None
        
        dates = data['DATE'].values
        forecast_dates = pd.date_range(start=data['DATE'].max() + timedelta(days=1), periods=forecast_steps)
        train_size = int(len(series) * 0.8)
        train_data = data.iloc[:train_size]
        val_data = data.iloc[train_size:] if train_size < len(series) else pd.DataFrame()
        
        mae, rmse = None, None
        
        if model_type == "LSTM":
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(series.reshape(-1, 1))
            scaled_train = scaler.transform(train_data[target_col].values.reshape(-1, 1))
            time_steps = min(5, len(scaled_train) - 1)
            X_train, y_train = [], []
            for i in range(time_steps, len(scaled_train)):
                X_train.append(scaled_train[i - time_steps:i, 0])
                y_train.append(scaled_train[i, 0])
            X_train, y_train = np.array(X_train), np.array(y_train)
            if len(X_train) > 0:
                X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(time_steps, 1)),
                    Dropout(0.2),
                    LSTM(25),
                    Dropout(0.2),
                    Dense(10),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(X_train, y_train, epochs=30, batch_size=8, validation_split=0.1, verbose=0)
                if not val_data.empty:
                    scaled_val = scaler.transform(val_data[target_col].values.reshape(-1, 1))
                    X_val, y_val = [], []
                    for i in range(time_steps, len(scaled_val)):
                        X_val.append(scaled_val[i - time_steps:i, 0])
                        y_val.append(scaled_val[i, 0])
                    if len(X_val) > 0:
                        X_val = np.array(X_val).reshape((len(X_val), time_steps, 1))
                        val_pred = scaler.inverse_transform(model.predict(X_val, verbose=0))
                        y_val = scaler.inverse_transform(np.array(y_val).reshape(-1, 1))
                        mae = mean_absolute_error(y_val, val_pred)
                        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                last_sequence = scaled_data[-time_steps:]
                predictions = []
                for _ in range(forecast_steps):
                    x_input = last_sequence.reshape((1, time_steps, 1))
                    pred = model.predict(x_input, verbose=0)
                    predictions.append(pred[0, 0])
                    last_sequence = np.append(last_sequence[1:], pred)
                predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
                return pd.DataFrame({'DATE': forecast_dates, 'PREDICTED': predictions.flatten()}), mae, rmse
        
        elif model_type == "Prophet":
            prophet_data = data[['DATE', target_col]].rename(columns={'DATE': 'ds', target_col: 'y'})
            train_data = prophet_data.iloc[:train_size]
            model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
            model.fit(train_data)
            if not val_data.empty:
                future_val = model.make_future_dataframe(periods=len(val_data), freq='D')
                forecast_val = model.predict(future_val)
                val_pred = forecast_val['yhat'].iloc[train_size:]
                mae = mean_absolute_error(val_data['y'], val_pred)
                rmse = np.sqrt(mean_squared_error(val_data['y'], val_pred))
            future = model.make_future_dataframe(periods=forecast_steps, freq='D')
            forecast = model.predict(future)
            predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[-forecast_steps:]
            return predictions.rename(columns={'ds': 'DATE', 'yhat': 'PREDICTED'}), mae, rmse
        
        elif model_type == "Linear Regression":
            X = np.arange(len(train_data)).reshape(-1, 1)
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            y = train_data[target_col]
            model = LinearRegression()
            model.fit(X_poly, y)
            if not val_data.empty:
                X_val = np.arange(len(train_data), len(train_data) + len(val_data)).reshape(-1, 1)
                X_val_poly = poly.transform(X_val)
                val_pred = model.predict(X_val_poly)
                mae = mean_absolute_error(val_data[target_col], val_pred)
                rmse = np.sqrt(mean_squared_error(val_data[target_col], val_pred))
            future_X = np.arange(len(data), len(data) + forecast_steps).reshape(-1, 1)
            future_X_poly = poly.transform(future_X)
            predictions = model.predict(future_X_poly)
            return pd.DataFrame({'DATE': forecast_dates, 'PREDICTED': predictions}), mae, rmse
        
        elif model_type == "XGBoost":
            features = ['DAY_OF_WEEK', 'TOTAL TRUCKS', 'TOTAL BUSES']
            X = data[features].fillna(data[features].median())
            y = data[target_col]
            X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, shuffle=False)
            model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            if not X_val.empty:
                val_pred = model.predict(X_val)
                mae = mean_absolute_error(y_val, val_pred)
                rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            last_features = X.iloc[-1:].copy()
            predictions = []
            for i in range(forecast_steps):
                pred = model.predict(last_features)
                predictions.append(pred[0])
                last_features['DAY_OF_WEEK'] = (last_features['DAY_OF_WEEK'] + 1) % 7
            return pd.DataFrame({'DATE': forecast_dates, 'PREDICTED': predictions}), mae, rmse
        
        elif model_type == "Prophet with Regressors":
            prophet_data = data[['DATE', target_col, 'DAY_OF_WEEK', 'TOTAL TRUCKS']].rename(
                columns={'DATE': 'ds', target_col: 'y'}
            ).fillna(data[['DAY_OF_WEEK', 'TOTAL TRUCKS']].median())
            train_data = prophet_data.iloc[:train_size]
            model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
            model.add_regressor('DAY_OF_WEEK')
            model.add_regressor('TOTAL TRUCKS')
            model.fit(train_data)
            if not val_data.empty:
                future_val = model.make_future_dataframe(periods=len(val_data), freq='D')
                future_val['DAY_OF_WEEK'] = val_data['DAY_OF_WEEK'].values
                future_val['TOTAL TRUCKS'] = val_data['TOTAL TRUCKS'].values
                forecast_val = model.predict(future_val)
                val_pred = forecast_val['yhat'].iloc[train_size:]
                mae = mean_absolute_error(val_data['y'], val_pred)
                rmse = np.sqrt(mean_squared_error(val_data['y'], val_pred))
            future = model.make_future_dataframe(periods=forecast_steps, freq='D')
            last_day = data['DAY_OF_WEEK'].iloc[-1]
            last_trucks = data['TOTAL TRUCKS'].iloc[-1]
            future['DAY_OF_WEEK'] = [(last_day + i) % 7 for i in range(len(future))]
            future['TOTAL TRUCKS'] = last_trucks
            forecast = model.predict(future)
            predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[-forecast_steps:]
            return predictions.rename(columns={'ds': 'DATE', 'yhat': 'PREDICTED'}), mae, rmse
        
        return pd.DataFrame(), None, None

    # Forecast Total Traffic
    st.write("#### Total Traffic Forecast")
    traffic_forecast, traffic_mae, traffic_rmse = forecast_data(station_data, 'TOTAL TRAFFIC', model_choice, forecast_steps)
    if not traffic_forecast.empty:
        combined_traffic = pd.concat([
            station_data[['DATE', 'TOTAL TRAFFIC']].rename(columns={'TOTAL TRAFFIC': 'Value'}),
            traffic_forecast.rename(columns={'PREDICTED': 'Value'})
        ])
        combined_traffic['Type'] = ['Historical'] * len(station_data) + ['Forecast'] * len(traffic_forecast)
        fig_traffic_forecast = px.line(combined_traffic, x='DATE', y='Value', color='Type', 
                                       title=f"Total Traffic Forecast for {selected_station} ({model_choice})")
        fig_traffic_forecast = apply_custom_plotly_layout(fig_traffic_forecast)
        if 'yhat_lower' in traffic_forecast.columns:
            fig_traffic_forecast.add_scatter(x=traffic_forecast['DATE'], y=traffic_forecast['yhat_lower'], 
                                             mode='lines', name='Lower Bound', line=dict(dash='dash', color='#d3d3d3'))
            fig_traffic_forecast.add_scatter(x=traffic_forecast['DATE'], y=traffic_forecast['yhat_upper'], 
                                             mode='lines', name='Upper Bound', line=dict(dash='dash', color='#d3d3d3'))
        fig_traffic_forecast.add_vline(x=station_data['DATE'].max(), line_dash="dash", line_color="#d3d3d3")
        st.plotly_chart(fig_traffic_forecast, use_container_width=True)
        with st.expander("Forecast Table"):
            st.dataframe(traffic_forecast)
            st.download_button("Download Forecast", traffic_forecast.to_csv(index=False), "traffic_forecast.csv", "text/csv")
        if traffic_mae is not None and traffic_rmse is not None:
            st.write(f"**Model Performance (Validation)**: MAE = {traffic_mae:.2f}, RMSE = {traffic_rmse:.2f}")
    else:
        st.write("No forecast data available.")

    # Forecast Total Overloads
    st.write("#### Total Overloads Forecast")
    overload_forecast, overload_mae, overload_rmse = forecast_data(station_data, 'TOTAL_OVERLOADS', model_choice, forecast_steps)
    if not overload_forecast.empty:
        combined_overloads = pd.concat([
            station_data[['DATE', 'TOTAL_OVERLOADS']].rename(columns={'TOTAL_OVERLOADS': 'Value'}),
            overload_forecast.rename(columns={'PREDICTED': 'Value'})
        ])
        combined_overloads['Type'] = ['Historical'] * len(station_data) + ['Forecast'] * len(overload_forecast)
        fig_overload_forecast = px.line(combined_overloads, x='DATE', y='Value', color='Type', 
                                        title=f"Total Overloads Forecast for {selected_station} ({model_choice})")
        fig_overload_forecast = apply_custom_plotly_layout(fig_overload_forecast)
        if 'yhat_lower' in overload_forecast.columns:
            fig_overload_forecast.add_scatter(x=overload_forecast['DATE'], y=overload_forecast['yhat_lower'], 
                                              mode='lines', name='Lower Bound', line=dict(dash='dash', color='#d3d3d3'))
            fig_overload_forecast.add_scatter(x=overload_forecast['DATE'], y=overload_forecast['yhat_upper'], 
                                              mode='lines', name='Upper Bound', line=dict(dash='dash', color='#d3d3d3'))
        fig_overload_forecast.add_vline(x=station_data['DATE'].max(), line_dash="dash", line_color="#d3d3d3")
        st.plotly_chart(fig_overload_forecast, use_container_width=True)
        with st.expander("Forecast Table"):
            st.dataframe(overload_forecast)
            st.download_button("Download Forecast", overload_forecast.to_csv(index=False), "overload_forecast.csv", "text/csv")
        if overload_mae is not None and overload_rmse is not None:
            st.write(f"**Model Performance (Validation)**: MAE = {overload_mae:.2f}, RMSE = {overload_rmse:.2f}")
    else:
        st.write("No forecast data available.")

# Tab 4: Prescriptive Analytics
with tab4:
    st.markdown("### Prescriptive Analytics")
    high_traffic = filtered_df.groupby('STATION')['TOTAL TRAFFIC'].mean().reset_index()
    threshold = high_traffic['TOTAL TRAFFIC'].quantile(0.75)
    busy_stations = high_traffic[high_traffic['TOTAL TRAFFIC'] > threshold]['STATION'].tolist()
    st.write("#### Recommendations")
    if busy_stations:
        st.write(f"Stations with high traffic (above {threshold:.0f}): {', '.join(busy_stations)}")
        st.write("- Consider increasing staff or resources at these stations.")
    else:
        st.write("No stations exceed the high-traffic threshold.")
    fig_busy = px.bar(high_traffic, x='STATION', y='TOTAL TRAFFIC', title="Average Traffic by Station")
    fig_busy = apply_custom_plotly_layout(fig_busy)
    st.plotly_chart(fig_busy, use_container_width=True)

# Tab 5: Diagnostic Analytics
with tab5:
    st.markdown("### Diagnostic Analytics")
    st.write("#### Traffic Trend Analysis")
    trend_fig = px.line(filtered_df, x='DATE', y='TOTAL TRAFFIC', title="Traffic Trend Over Time")
    trend_fig = apply_custom_plotly_layout(trend_fig)
    st.plotly_chart(trend_fig, use_container_width=True)
    
    with st.expander("Outlier Detection"):
        q1 = filtered_df['TOTAL TRAFFIC'].quantile(0.25)
        q3 = filtered_df['TOTAL TRAFFIC'].quantile(0.75)
        iqr = q3 - q1
        outliers = filtered_df[(filtered_df['TOTAL TRAFFIC'] < q1 - 1.5 * iqr) | (filtered_df['TOTAL TRAFFIC'] > q3 + 1.5 * iqr)]
        st.dataframe(outliers)
        fig_outliers = px.scatter(outliers, x='DATE', y='TOTAL TRAFFIC', color='STATION', title="Traffic Outliers")
        fig_outliers = apply_custom_plotly_layout(fig_outliers)
        st.plotly_chart(fig_outliers, use_container_width=True)

    st.write("#### Key Influencers of Unknowns")
    correlation_df = df[['UNKNOWN', 'PASSENGER CAR', 'TOTAL TRUCKS', 'TOTAL BUSES', 'TOTAL TRAFFIC']].corr()
    fig_corr = px.imshow(correlation_df, text_auto=True, title="Correlation Heatmap", color_continuous_scale='RdBu_r')
    fig_corr = apply_custom_plotly_layout(fig_corr)
    st.plotly_chart(fig_corr, use_container_width=True)
    st.write("**Analysis**: High positive correlations suggest categories that increase unknowns; negative correlations may reduce them.")

# CSS Styling with Professional Gradient Background
st.markdown("""
    <style>
    /* Main app background with a professional gradient */
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: #ffffff;
        font-family: 'Arial', sans-serif;
    }

    /* Ensure all text elements are readable against the dark gradient */
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #ffffff !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
    }

    /* Style for tabs */
    .stTabs [role="tablist"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 5px;
    }

    .stTabs [role="tab"] {
        color: #ffffff;
        border-radius: 5px;
        padding: 8px 15px;
        transition: background-color 0.3s;
    }

    .stTabs [role="tab"][aria-selected="true"] {
        background-color: #ffffff;
        color: #1e3c72 !important;
        font-weight: bold;
    }

    /* Style for select boxes, date inputs, and buttons */
    .stSelectbox, .stDateInput, .stMultiselect, .stButton>button {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 8px;
        padding: 8px;
        color: #1e3c72 !important;
        border: 1px solid #2a5298;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }

    .stSelectbox:hover, .stDateInput:hover, .stMultiselect:hover, .stButton>button:hover {
        background-color: #ffffff;
        border-color: #1e3c72;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    /* Style for expanders */
    .stExpander {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 0.2);
    }

    .stExpander summary {
        color: #ffffff;
        font-weight: bold;
    }

    /* Style for dataframes */
    .stDataFrame {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 8px;
        padding: 10px;
        color: #1e3c72;
    }

    /* Style for metrics */
    .stMetric {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .stMetric label {
        color: #ffffff !important;
        font-weight: bold;
    }

    .stMetric div {
        color: #ffffff !important;
    }

    /* Remove the CSS override for Plotly chart background since we're handling it in Plotly's layout */
    </style>
""", unsafe_allow_html=True)