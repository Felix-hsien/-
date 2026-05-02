import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="物理實驗數據分析工具", layout="wide")
st.title("📊 阻尼振盪：上包絡偵測與衰減分析")

# 檔案上傳
uploaded_file = st.file_uploader("請上傳 CSV 檔案 (格式：time, displacement)", type="csv")

if uploaded_file is not None:
    # 讀取數據，支援你提供的 time, displacement 格式
    df = pd.read_csv(uploaded_file)
    
    # 清洗數據：確保欄位名稱正確（移除多餘空格）
    df.columns = [c.strip().lower() for c in df.columns]
    
    if 'time' in df.columns and 'displacement' in df.columns:
        t = df['time'].values
        y = df['displacement'].values

        # 1. 偵測上包絡峰值
        # 你的數據約每 0.33 秒一個波峰，採樣間隔約 0.033，故 distance 設為 5~10 較合適
        peaks, _ = find_peaks(y, distance=8, prominence=0.001)
        t_peaks = t[peaks]
        y_peaks = y[peaks]

        # 介面佈局
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("位移 vs 時間 (原始數據與峰值)")
            fig1, ax1 = plt.subplots()
            ax1.plot(t, y, label='震盪數據', alpha=0.6, color='steelblue')
            ax1.scatter(t_peaks, y_peaks, color='red', s=25, label='偵測到的峰值')
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Displacement (m)")
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig1)

        with col2:
            st.subheader("ln(位移) vs 時間 (線性回歸)")
            
            # 過濾正值峰值以計算 ln
            mask = y_peaks > 0
            t_fit = t_peaks[mask]
            ln_y_fit = np.log(y_peaks[mask])

            if len(t_fit) > 1:
                # 線性回歸
                X = t_fit.reshape(-1, 1)
                model = LinearRegression().fit(X, ln_y_fit)
                ln_y_pred = model.predict(X)
                r2 = model.score(X, ln_y_fit)

                fig2, ax2 = plt.subplots()
                ax2.scatter(t_fit, ln_y_fit, color='red', label='ln(峰值數據)')
                ax2.plot(t_fit, ln_y_pred, color='navy', linestyle='-', label='回歸預測線')
                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("ln(Displacement)")
                ax2.legend()
                ax2.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig2)

                # 顯示物理參數
                st.info(f"### 物理分析結果")
                st.latex(rf"\ln(A) = {model.coef_[0]:.4f} \cdot t + {model.intercept_:.4f}")
                st.write(f"**衰減係數 ($\gamma$):** {-model.coef_[0]:.6f}")
                st.write(f"**判定係數 ($R^2$):** {r2:.6f}")
            else:
                st.error("有效峰值不足，請調整偵測參數。")

        # 顯示數據表
        with st.expander("查看偵測到的峰值數據表"):
            peak_df = pd.DataFrame({'Time': t_fit, 'Displacement': y_peaks[mask], 'ln(Disp)': ln_y_fit})
            st.write(peak_df)

    else:
        st.error("CSV 格式不符！請確保標題包含 'time' 與 'displacement'。")
else:
    st.write("請從上方上傳你剛才提供的 CSV 數據。")
