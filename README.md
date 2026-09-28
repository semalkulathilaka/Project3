# 🛒 Walmart Store Sales Analytics & Forecasting Dashboard

An end-to-end Machine Learning and Exploratory Data Analysis (EDA) dashboard built with **Streamlit**, **Scikit-Learn**, and **XGBoost**. This interactive application enables retail decision-makers to analyze historical sales trends, benchmark predictive algorithms, forecast future demand (2013 simulation), and simulate "What-If" scenarios driven by macroeconomic indicators.



## 📌 Features

**📊 Exploratory Data Analysis (EDA):**
   Summary metrics: total sales, store counts, date spans, and weekly averages.
   Interactive data tables with descriptive statistics.
   Holiday vs. non-holiday sales distribution comparison.

 **📈 Seasonal & Store Trends:**
   Interactive time-series trends tracking monthly sales across all locations.
   Ranked bar charts visualizing sales contributions per store.
   Year-over-year seasonal comparisons (2011 vs. 2012).

**🤖 Multi-Model Machine Learning Benchmark:**
   Compares **Random Forest**, **Gradient Boosting**, and **XGBoost Regressors**.
   Computes standard evaluation metrics: $R^2$ Score, Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).
   Actual vs. Predicted sales comparison charts with configurable test splits.

  **🔮 2013 Sales Forecasting Simulation:**
   Projects monthly sales across all 45 stores into 2013 based on macroeconomic baselines.
   Multi-line comparative visualization contrasting actual historical sales (2010–2012) with 2013 predictions.

 **🎯 Interactive "What-If" Predictor:**
   Interactive simulator allowing users to tweak Store ID, Month, Fuel Price, Temperature, CPI, and Unemployment rates to estimate weekly sales in real time.



## 🛠️ Tech Stack & Libraries

 **Language:** Python 3.10+
 **Frontend / Dashboard:** [Streamlit](https://streamlit.io/)
 **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
 **Visualization:** [Plotly](https://plotly.com/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
 **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/), [XGBoost](https://xgboost.ai/)



## 📂 Project Structure

├── app.py                      # Main Streamlit dashboard application
├── Walmart_Store_sales.csv     # Historical Walmart sales dataset
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation

link -> https://project3-j98tdv55eb349uccbnjfed.streamlit.app/
