import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

# 1. Page Configuration
st.set_page_config(page_title="Walmart Sales Analysis & Forecasting", layout="wide", page_icon="🛒")

st.title("🛒 Walmart Store Sales Analytics & Prediction Dashboard")

# 2. Locate Dataset
DATA_PATH = "Walmart_Store_sales.csv"

if not os.path.exists(DATA_PATH):
    fallback = os.path.join(os.path.dirname(__file__), "Walmart_Store_sales.csv")
    if os.path.exists(fallback):
        DATA_PATH = fallback
    elif os.path.exists(os.path.expanduser(r"~\Downloads\Walmart_Store_sales.csv")):
        DATA_PATH = os.path.expanduser(r"~\Downloads\Walmart_Store_sales.csv")

@st.cache_data
def load_data(file_path):
    # Use dayfirst=True for dates formatted like DD-MM-YYYY
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Month_Year'] = df['Date'].dt.to_period('M').astype(str)
    return df

# Sidebar file uploader or default dataset
uploaded_file = st.sidebar.file_uploader("Upload Walmart Sales CSV", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
elif os.path.exists(DATA_PATH):
    df = load_data(DATA_PATH)
else:
    st.error("Walmart_Store_sales.csv not found. Please upload it via the sidebar.")
    st.stop()

# 3. Sidebar Navigation
page = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 Data Overview & EDA",
        "📈 Seasonal & Store Trends",
        "🤖 Model Comparison & Evaluation",
        "🔮 2013 Sales Forecast",
        "🎯 Custom What-If Predictor"
    ]
)

# ----------------- PAGE 1: EDA -----------------
if page == "📊 Data Overview & EDA":
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Total Stores", df['Store'].nunique())
    col3.metric("Date Range", f"{df['Date'].min().strftime('%d %b %Y')} - {df['Date'].max().strftime('%d %b %Y')}")
    col4.metric("Average Weekly Sales", f"${df['Weekly_Sales'].mean():,.2f}")

    st.subheader("Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe().T, use_container_width=True)

    st.subheader("Weekly Sales Distribution: Holiday vs Non-Holiday")
    fig, ax = plt.subplots(figsize=(6, 4))
    holiday_sales = df.groupby('Holiday_Flag')['Weekly_Sales'].mean()
    holiday_sales.plot(kind='bar', color=['skyblue', 'orange'], ax=ax)
    ax.set_title("Average Weekly Sales: Holiday vs Non-Holiday")
    ax.set_xlabel("Holiday Flag")
    ax.set_ylabel("Average Weekly Sales ($)")
    ax.set_xticklabels(['Non-Holiday (0)', 'Holiday (1)'], rotation=0)
    st.pyplot(fig)

# ----------------- PAGE 2: TRENDS -----------------
elif page == "📈 Seasonal & Store Trends":
    st.header("📈 Sales Trends & Patterns")

    tab1, tab2, tab3 = st.tabs(["Monthly Overall Trend", "Total Sales by Store", "Seasonal 2011 vs 2012 Comparison"])

    with tab1:
        st.subheader("Monthly Walmart Sales Trend")
        monthly_sales = df.groupby('Month_Year')['Weekly_Sales'].sum().reset_index()
        fig_month = px.line(
            monthly_sales, 
            x='Month_Year', 
            y='Weekly_Sales', 
            markers=True, 
            title="Total Monthly Sales Trend (All Stores)",
            labels={'Month_Year': 'Month-Year', 'Weekly_Sales': 'Total Weekly Sales ($)'}
        )
        st.plotly_chart(fig_month, use_container_width=True)

    with tab2:
        st.subheader("Total Sales by Store")
        store_sales = df.groupby('Store')['Weekly_Sales'].sum().sort_values(ascending=False).reset_index()
        fig_store = px.bar(
            store_sales, 
            x='Store', 
            y='Weekly_Sales', 
            title="Total Sales by Store Ranked",
            labels={'Store': 'Store ID', 'Weekly_Sales': 'Total Sales ($)'},
            color='Weekly_Sales',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_store, use_container_width=True)

    with tab3:
        st.subheader("Seasonal Trend of Walmart Sales (2011 vs 2012)")
        data_filtered = df[df['Year'].isin([2011, 2012])]
        seasonal_sales = data_filtered.groupby(['Year', 'Month'])['Weekly_Sales'].sum().reset_index()
        pivot_sales = seasonal_sales.pivot(index='Month', columns='Year', values='Weekly_Sales')
        
        fig, ax = plt.subplots(figsize=(10, 5))
        pivot_sales.plot(kind='line', marker='o', ax=ax)
        ax.set_title("Seasonal Trend: 2011 vs 2012")
        ax.set_xlabel("Month")
        ax.set_ylabel("Total Weekly Sales ($)")
        ax.set_xticks(range(1, 13))
        ax.grid(True)
        st.pyplot(fig)

# ----------------- PAGE 3: MODEL BENCHMARKS -----------------
elif page == "🤖 Model Comparison & Evaluation":
    st.header("🤖 Machine Learning Model Benchmarks")
    st.write("Train Random Forest, Gradient Boosting, and XGBoost on historical data (2010–2012) and evaluate performance.")

    feature_cols = ['Store', 'Month', 'Year', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Holiday_Flag']
    X = df[feature_cols]
    y = df['Weekly_Sales']

    test_size = st.sidebar.slider("Test Set Split Ratio", 0.1, 0.4, 0.2, 0.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    with st.spinner("Training models (Random Forest, Gradient Boosting, XGBoost)..."):
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)

        gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        gb.fit(X_train, y_train)
        gb_preds = gb.predict(X_test)

        xgb_mod = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        xgb_mod.fit(X_train, y_train)
        xgb_preds = xgb_mod.predict(X_test)

    metrics_data = {
        "Model": ["Random Forest", "Gradient Boosting", "XGBoost"],
        "R² Score": [r2_score(y_test, rf_preds), r2_score(y_test, gb_preds), r2_score(y_test, xgb_preds)],
        "RMSE ($)": [np.sqrt(mean_squared_error(y_test, rf_preds)), np.sqrt(mean_squared_error(y_test, gb_preds)), np.sqrt(mean_squared_error(y_test, xgb_preds))],
        "MAE ($)": [mean_absolute_error(y_test, rf_preds), mean_absolute_error(y_test, gb_preds), mean_absolute_error(y_test, xgb_preds)]
    }
    metrics_df = pd.DataFrame(metrics_data)

    st.subheader("Performance Metrics")
    col1, col2, col3 = st.columns(3)
    best_r2 = metrics_df.loc[metrics_df['R² Score'].idxmax()]
    col1.metric("Top Model (R²)", f"{best_r2['Model']}", f"{best_r2['R² Score']:.4f}")
    best_rmse = metrics_df.loc[metrics_df['RMSE ($)'].idxmin()]
    col2.metric("Lowest RMSE", f"{best_rmse['Model']}", f"${best_rmse['RMSE ($)']:,.2f}")
    best_mae = metrics_df.loc[metrics_df['MAE ($)'].idxmin()]
    col3.metric("Lowest MAE", f"{best_mae['Model']}", f"${best_mae['MAE ($)']:,.2f}")

    st.table(metrics_df.style.format({"R² Score": "{:.4f}", "RMSE ($)": "${:,.2f}", "MAE ($)": "${:,.2f}"}))

    st.subheader("Actual vs Predicted Sales")
    num_samples = st.slider("Sample points to display", 20, 200, 100, 10)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(y_test.values[:num_samples], label="Actual Sales", color="black", linewidth=2)
    ax.plot(rf_preds[:num_samples], label=f"Random Forest (R²={r2_score(y_test, rf_preds):.2f})", color="green", linestyle="--")
    ax.plot(gb_preds[:num_samples], label=f"Gradient Boosting (R²={r2_score(y_test, gb_preds):.2f})", color="blue", linestyle="-.")
    ax.plot(xgb_preds[:num_samples], label=f"XGBoost (R²={r2_score(y_test, xgb_preds):.2f})", color="red", linestyle=":")
    ax.set_title(f"Actual vs Predicted Weekly Sales (Sample {num_samples} points)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Weekly Sales ($)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# ----------------- PAGE 4: 2013 FORECAST -----------------
elif page == "🔮 2013 Sales Forecast":
    st.header("🔮 2013 Walmart Sales Simulation & Forecast")
    st.write("Generates 2013 simulation features across all stores and predicts monthly totals using models trained on 2010–2012.")

    train_df = df[df['Year'] <= 2012]
    features = ['Store', 'Month', 'Year', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Holiday_Flag']
    X_train = train_df[features]
    y_train = train_df['Weekly_Sales']

    with st.spinner("Fitting models and forecasting 2013..."):
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1).fit(X_train, y_train)
        gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42).fit(X_train, y_train)
        xgb_m = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42).fit(X_train, y_train)

        months = range(1, 13)
        stores = df['Store'].unique()
        rows = []
        for store in stores:
            for month in months:
                rows.append({
                    'Store': store,
                    'Month': month,
                    'Year': 2013,
                    'Temperature': df['Temperature'].mean(),
                    'Fuel_Price': df['Fuel_Price'].mean(),
                    'CPI': df['CPI'].mean(),
                    'Unemployment': df['Unemployment'].mean(),
                    'Holiday_Flag': 1 if month in [1, 11, 12] else 0
                })
        test_2013 = pd.DataFrame(rows)
        test_2013['RF_Pred'] = rf.predict(test_2013[features])
        test_2013['GB_Pred'] = gb.predict(test_2013[features])
        test_2013['XGB_Pred'] = xgb_m.predict(test_2013[features])

    monthly_sales = train_df.groupby(['Year', 'Month'])['Weekly_Sales'].sum().reset_index()
    pivot_sales = monthly_sales.pivot(index='Month', columns='Year', values='Weekly_Sales')

    rf_pred_2013 = test_2013.groupby('Month')['RF_Pred'].sum()
    gb_pred_2013 = test_2013.groupby('Month')['GB_Pred'].sum()
    xgb_pred_2013 = test_2013.groupby('Month')['XGB_Pred'].sum()

    forecast_fig = go.Figure()
    if 2010 in pivot_sales.columns:
        forecast_fig.add_trace(go.Scatter(x=pivot_sales.index, y=pivot_sales[2010], mode='lines+markers', name='2010 Actual', line=dict(color='blue')))
    if 2011 in pivot_sales.columns:
        forecast_fig.add_trace(go.Scatter(x=pivot_sales.index, y=pivot_sales[2011], mode='lines+markers', name='2011 Actual', line=dict(color='green')))
    if 2012 in pivot_sales.columns:
        forecast_fig.add_trace(go.Scatter(x=pivot_sales.index, y=pivot_sales[2012], mode='lines+markers', name='2012 Actual', line=dict(color='red')))

    forecast_fig.add_trace(go.Scatter(x=rf_pred_2013.index, y=rf_pred_2013.values, mode='lines+markers', name='2013 RF Predicted', line=dict(color='orange', dash='dash')))
    forecast_fig.add_trace(go.Scatter(x=gb_pred_2013.index, y=gb_pred_2013.values, mode='lines+markers', name='2013 GB Predicted', line=dict(color='purple', dash='dot')))
    forecast_fig.add_trace(go.Scatter(x=xgb_pred_2013.index, y=xgb_pred_2013.values, mode='lines+markers', name='2013 XGB Predicted', line=dict(color='brown', dash='dashdot')))

    forecast_fig.update_layout(
        title="Seasonal Monthly Sales Trend (2010–2013: Actual vs Predicted)",
        xaxis_title="Month",
        yaxis_title="Total Monthly Sales ($)",
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        template="plotly_white"
    )
    st.plotly_chart(forecast_fig, use_container_width=True)

# ----------------- PAGE 5: WHAT-IF PREDICTOR -----------------
elif page == "🎯 Custom What-If Predictor":
    st.header("🎯 Single-Store What-If Prediction")
    st.write("Select store and macroeconomic factors to predict weekly sales in real time.")

    features = ['Store', 'Month', 'Year', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Holiday_Flag']
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(df[features], df['Weekly_Sales'])

    col1, col2 = st.columns(2)
    with col1:
        store_val = st.selectbox("Store ID", sorted(df['Store'].unique()))
        month_val = st.slider("Month", 1, 12, 6)
        year_val = st.selectbox("Year", [2012, 2013, 2014])
        holiday_val = st.selectbox("Holiday Week?", [0, 1], format_func=lambda x: "Yes (Holiday)" if x == 1 else "No (Regular)")

    with col2:
        temp_val = st.slider("Temperature (°F)", float(df['Temperature'].min()), float(df['Temperature'].max()), float(df['Temperature'].mean()))
        fuel_val = st.slider("Fuel Price ($)", float(df['Fuel_Price'].min()), float(df['Fuel_Price'].max()), float(df['Fuel_Price'].mean()))
        cpi_val = st.slider("CPI", float(df['CPI'].min()), float(df['CPI'].max()), float(df['CPI'].mean()))
        unemp_val = st.slider("Unemployment Rate (%)", float(df['Unemployment'].min()), float(df['Unemployment'].max()), float(df['Unemployment'].mean()))

    input_df = pd.DataFrame([{
        'Store': store_val,
        'Month': month_val,
        'Year': year_val,
        'Temperature': temp_val,
        'Fuel_Price': fuel_val,
        'CPI': cpi_val,
        'Unemployment': unemp_val,
        'Holiday_Flag': holiday_val
    }])

    pred = rf_model.predict(input_df)[0]
    st.success(f"### 🏷️ Estimated Weekly Sales: **${pred:,.2f}**")