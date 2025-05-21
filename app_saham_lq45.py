# Import library
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from streamlit_option_menu import option_menu
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import plotly.graph_objects as go
import yfinance as yf
import time
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings

warnings.filterwarnings("ignore")

def ensure_float_dataframe(df):
    return df.apply(pd.to_numeric, errors='coerce')


# Import fungsi load data
from data_loader import load_and_update_data

# Konfigurasi halaman
st.set_page_config(page_title="PREDIKSI STOCK PRICE LQ45", layout="wide")

TICKERS = [
    'ACES', 'ADMR', 'ADRO', 'AKRA', 'AMMN',
    'AMRT', 'ANTM', 'ARTO', 'ASII', 'BBCA',
    'BBNI', 'BBRI', 'BBTN', 'BMRI', 'BRIS',
    'BRPT', 'CPIN', 'CTRA', 'ESSA', 'EXCL',
    'GOTO', 'ICBP', 'INCO', 'INDF', 'INKP',
    'ISAT', 'ITMG', 'JPFA', 'JSMR', 'KLBF',
    'MAPA', 'MAPI', 'MDMA', 'MDKA', 'MEDC',
    'PGAS', 'PGEO', 'PTBA', 'SIDO', 'SMGR',
    'SMRA', 'TLKM', 'TOWR', 'UNTR', 'UNVR'
]
# Sidebar untuk konfigurasi
st.sidebar.title("KONFIGURASI DATA")
lq45_symbols = [s.replace(".csv", "") for s in os.listdir("saham_lq45")]
selected_symbol = st.sidebar.selectbox("Pilih simbol saham:", sorted(TICKERS))
start_date = st.sidebar.date_input("Tanggal mulai", datetime(2024, 1, 1))
end_date = st.sidebar.date_input("Tanggal akhir", datetime.today())
model_type = st.sidebar.selectbox("Pilih Model", ["GLM", "ARIMA"])
theme_option = st.sidebar.selectbox("Pilih Tema", ["Terang", "Gelap"])
st.sidebar.title("Download Data Saham Baru")
if st.sidebar.button("Download Data Terbaru"):
    try:
        stock_data = yf.download(f"{selected_symbol}.JK", start=start_date, end=end_date)
        if not os.path.exists("saham_lq45_new"):
            os.makedirs("saham_lq45_new")
        stock_data.reset_index(inplace=True)
        stock_data.rename(columns={
            "Date": "date",
            "Open": "open_price",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }, inplace=True)
        stock_data.to_csv(f"saham_lq45_new/{selected_symbol}.csv", index=False)
        st.sidebar.success(f"Data {selected_symbol} berhasil disimpan di folder saham_lq45_new!")
    except Exception as e:
        st.sidebar.error(f"Terjadi kesalahan: {e}")

# Tema warna
if theme_option == "Gelap":
    bg_color, text_color = "#2c2c2c", "#f8f9fa"
    line_color_actual, line_color_pred = "#e63946", "#a8dadc"
else:
    bg_color, text_color = "#ffffff", "#000000"
    line_color_actual, line_color_pred = "blue", "red"


#FUNGSI tuning GLM DAN ARIMA
def create_glm_features(data, lags=5):
    df_feat = pd.DataFrame()
    for i in range(1, lags + 1):
        df_feat[f"lag_{i}"] = data.shift(i)
    df_feat["target"] = data.values
    return df_feat.dropna()

def evaluate(true, pred):
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true, pred)
    mape = np.mean(np.abs((true - pred) / true)) * 100
    r2 = r2_score(true, pred)
    n, p = len(true), 1
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return mse, rmse, mae, mape, r2, adj_r2

def plot_error_visuals(true, pred):
    residuals = true - pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(residuals, bins=30, kde=True, ax=ax1, color='orange')
    ax1.set_title("Distribusi Error")
    ax1.set_xlabel("Error")
    ax1.set_ylabel("Frekuensi")
    corr_matrix = np.corrcoef([true, pred])
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm",
                xticklabels=["Aktual", "Prediksi"],
                yticklabels=["Aktual", "Prediksi"], ax=ax2)
    ax2.set_title("Heatmap Korelasi")
    st.pyplot(fig)

def tune_glm_model(data, max_lags=10):
    best_mse = float('inf')
    best_lags = 1
    best_model = None

    for lags in range(1, max_lags + 1):
        features = create_glm_features(data, lags=lags)
        features = features.dropna()

        if features.empty:
            continue

        X = features.drop("target", axis=1)
        y = features["target"]

        X, y = X.align(y, join='inner', axis=0)

        X = sm.add_constant(X, has_constant='add')

        X = X.astype(float)
        y = y.astype(float)

        model = sm.GLM(y, X).fit()
        pred = model.predict(X)
        mse = mean_squared_error(y, pred)

        if mse < best_mse:
            best_mse = mse
            best_lags = lags
            best_model = model

    return best_model, best_lags

def tune_arima_model(data, max_p=5, max_q=5):
    # Cek dulu: kalau masih pandas, baru dropna
    if isinstance(data, (pd.Series, pd.DataFrame)):
        data = data.dropna()

    # Pastikan jadi numpy array float
    data = np.array(data, dtype=float)

    best_aic = float('inf')
    best_order = None
    best_model = None

    total_iterations = (max_p) * (max_q)
    progress = st.progress(0)
    iteration = 0

    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            iteration += 1
            try:
                model = ARIMA(data, order=(p, 1, q))
                model_fit = model.fit()
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_order = (p, 1, q)
                    best_model = model_fit
            except Exception as e:
                st.warning(f"ARIMA({p},1,{q}) gagal: {e}")
            finally:
                progress.progress(min(iteration / total_iterations, 1.0))
                time.sleep(0.01)

    progress.empty()

    if best_model is None:
        st.error("Gagal menemukan model ARIMA terbaik. Silakan cek data atau ubah parameter pencarian.")
    else:
        st.success(f"Model terbaik ditemukan: ARIMA{best_order} dengan AIC: {best_aic:.2f}")

    return best_model, best_order

#Tulisan berjalan
st.markdown(
    """
    <marquee behavior="scroll" direction="left" style="color:red; font-size:20px; font-weight:bold;">
         Selamat datang di Aplikasi Prediksi Harga Saham LQ45 menggunakan Algoritma GLM dan ARIMA
    </marquee>
    """,
    unsafe_allow_html=True
)

#menu navigasi
selected = option_menu(
    menu_title="Navigasi",
    options=["Beranda", "Konfigurasi Data", "Prediksi Saham", "Evaluasi Model"],
    icons=["house", "cog", "chart-line", "bar-chart"],
    orientation="horizontal",
)

#halaman beranda
if selected == "Beranda":
    st.title("PREDIKSI HARGA SAHAM LQ45")
    st.markdown(f"Data Historis diambil dari yahoo finance sampai tanggal 21 Mei 2025.")

#Halaman konfigurasi data
elif selected == "Konfigurasi Data":
    st.title("KONFIGURASI DATA")
    st.write(f"Simbol Saham: **{selected_symbol}**")
    st.write(f"Rentang Tanggal: {start_date} hingga {end_date}")
    st.write(f"Model yang digunakan: **{model_type}**")

#halaman prediksi
elif selected == "Prediksi Saham":
    st.title("PREDIKSI HARGA SAHAM")

    df = load_and_update_data(selected_symbol)
    df_filtered = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))].copy()

    if df_filtered.empty:
        st.error("Data kosong untuk rentang tanggal tersebut.")
        st.stop()

    close_prices = df_filtered["close"].dropna().reset_index(drop=True)

    st.subheader("HISTORI HARGA SAHAM")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["open_price"], name='Open', line=dict(color='skyblue')))
    fig.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["high"], name='High', line=dict(color='limegreen')))
    fig.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["low"], name='Low', line=dict(color='salmon')))
    fig.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["close"], name='Close', line=dict(color=line_color_actual)))
    fig.update_layout(title="Harga Open, High, Low, Close", xaxis_title="Tanggal", yaxis_title="Harga", template="plotly_dark" if theme_option == "Gelap" else "plotly_white")
    st.plotly_chart(fig, use_container_width=True)


    def plot_forecast(actual_series, forecast_value):
        plt.figure(figsize=(10, 5))
        plt.plot(actual_series.index, actual_series.values, label="Data Aktual")
        # Tambahkan titik prediksi sebagai marker
        plt.scatter(actual_series.index[-1] + 1, forecast_value, color='red', label="Prediksi Berikutnya")
        plt.title("Harga Saham: Aktual dan Prediksi Berikutnya")
        plt.xlabel("Waktu")
        plt.ylabel("Harga")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

    if model_type == "GLM":
        st.subheader("Model: Generalized Linear Model (GLM)")
        best_model, best_lags = tune_glm_model(close_prices)
        st.write(f"Model terbaik dengan {best_lags} lag.")
        glm_summary = best_model.summary().as_text()
        st.code(glm_summary)

        features = create_glm_features(close_prices, lags=best_lags)
        features = features.dropna()

        if not features.empty:
            X = features.drop("target", axis=1)
            y = features["target"]
            X = sm.add_constant(X, has_constant='add')

            X = X.astype(float)
            pred = best_model.predict(X)

            st.subheader("Prediksi Harga ke Depan")
            n_days = st.number_input("Jumlah hari prediksi:", 1, 30, 3)

            window = close_prices.iloc[-best_lags:].tolist()
            future_preds = []

            for _ in range(n_days):
                input_vals = window[-best_lags:][::-1]  # urutkan mundur
                df_input = pd.DataFrame([input_vals], columns=[f"lag_{i}" for i in range(1, best_lags + 1)])
                df_input = sm.add_constant(df_input, has_constant='add')
                df_input = df_input.astype(float)

                pred_price = best_model.predict(df_input)[0]
                future_preds.append(pred_price)
                window.append(pred_price)

            for i, price in enumerate(future_preds, 1):
                st.success(f"Hari ke-{i}: {price:,.2f}")



    elif model_type == "ARIMA":

        st.subheader("Model: ARIMA")
        # Konversi dan bersihkan data harga
        close_prices = pd.to_numeric(close_prices, errors='coerce').dropna()
        # Uji stasioneritas menggunakan ADF Test
        adf_result = adfuller(close_prices)
        st.write("### Hasil Uji ADF (Augmented Dickey-Fuller):")
        st.write(f"ADF Statistic: {adf_result[0]:.4f}")
        st.write(f"p-value: {adf_result[1]:.4f}")

        # Tentukan apakah perlu differencing
        if adf_result[1] <= 0.05:
            st.warning("Data tidak stasioner (p ≥ 0.05). Melakukan differencing sebanyak 1 kali.")
            close_prices_diff = close_prices.diff().dropna()
            d = 1
        else:
            st.success("Data stasioner (p > 0.05). Tidak perlu differencing.")
            close_prices_diff = close_prices.copy()
            d = 0


        def tune_arima_model(data):
            best_aic = float('inf')
            best_order = None
            best_model = None
            for p in range(5):
                for q in range(5):
                    try:
                        model = ARIMA(data, order=(p, d, q)).fit()
                        if model.aic < best_aic:
                            best_aic = model.aic
                            best_order = (p, q)
                            best_model = model
                    except:
                        continue
            return best_model, best_order
        # Cari model terbaik
        best_model, best_order = tune_arima_model(close_prices_diff)
        if best_model is not None:
            st.success(f"Model terbaik ditemukan: ARIMA({best_order[0]}, {d}, {best_order[1]})")
            st.code(best_model.summary().as_text())
            # Prediksi harga berikutnya
            next_pred = best_model.forecast(steps=1)
            # Rekonstruksi prediksi jika dilakukan differencing
            if d == 1:
                last_actual = close_prices.iloc[-1]
                predicted_price = last_actual + next_pred.iloc[0]
            else:
                predicted_price = next_pred.iloc[0]
            st.metric(label="Prediksi Harga Berikutnya", value=f"{predicted_price:,.2f}")
        else:
            st.error("Model ARIMA terbaik tidak ditemukan. Silakan periksa data atau ubah parameter pencarian.")

#halaman evaluasi
elif selected == "Evaluasi Model":
    st.title("EVALUASI MODEL")

    # Load data
    df = load_and_update_data(selected_symbol)
    df_filtered = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))].copy()

    # Ambil harga penutupan
    close_prices = df_filtered["close"].dropna().reset_index(drop=True)

    if model_type == "GLM":
        st.subheader("Model: Generalized Linear Model (GLM)")

        # Tune model
        best_model, best_lags = tune_glm_model(close_prices)
        st.success(f"Model terbaik ditemukan dengan {best_lags} lag.")

        # Buat fitur
        features = create_glm_features(close_prices, lags=best_lags)
        X = sm.add_constant(features.drop(columns="target"), has_constant='add')
        X = ensure_float_dataframe(X)  # pastikan float
        y = features["target"]

        # Prediksi
        pred = best_model.predict(X)

        # Evaluasi
        y_true = np.array(y, dtype=float)
        y_pred = np.array(pred, dtype=float)

        mse, rmse, mae, mape, r2, adj_r2 = evaluate(y_true, y_pred)

        st.subheader("Hasil Evaluasi Model GLM")
        st.write(pd.DataFrame({
            "Metrik": ["MSE", "RMSE", "MAE", "MAPE", "R-squared", "Adjusted R-squared"],
            "Nilai": [mse, rmse, mae, mape, r2, adj_r2]
        }))

        st.subheader("Visualisasi Prediksi vs Aktual")
        plot_error_visuals(y_true, y_pred)

    elif model_type == "ARIMA":
        st.subheader("Model: ARIMA")

        close_prices = pd.to_numeric(close_prices, errors='coerce')
        close_prices = np.array(close_prices, dtype=float)

        # Tune model ARIMA
        best_model, best_order = tune_arima_model(close_prices)

        if best_model is not None:
            st.success(f"Model terbaik: ARIMA{best_order}")

            # Prediksi
            pred = best_model.predict(start=0, end=len(close_prices) - 1)

            y_true = np.array(close_prices[-len(pred):], dtype=float)
            y_pred = np.array(pred, dtype=float)

            # Evaluasi
            mse, rmse, mae, mape, r2, adj_r2 = evaluate(y_true, y_pred)

            st.subheader("Hasil Evaluasi Model ARIMA")
            st.write(pd.DataFrame({
                "Metrik": ["MSE", "RMSE", "MAE", "MAPE", "R-squared", "Adjusted R-squared"],
                "Nilai": [mse, rmse, mae, mape, r2, adj_r2]
            }))

            st.subheader("Visualisasi Prediksi vs Aktual")
            plot_error_visuals(y_true, y_pred)

        else:
            st.error("Model ARIMA terbaik tidak ditemukan. Silakan cek data atau parameter.")
