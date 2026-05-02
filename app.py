import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression

st.title("上包絡偵測與回歸分析程式")

# 1. 檔案上傳
uploaded_file = st.file_uploader("請上傳 CSV 數據檔案", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 上傳的資料預覽", df.head())
    
    # 讓使用者選擇欄位
    columns = df.columns.tolist()
    time_col = st.selectbox("請選擇『時間』欄位", columns)
    val_col = st.selectbox("請選擇『位移/振幅』欄位", columns)
    
    t = df[time_col].values
    y = df[val_col].values

    # 2. 尋找波峰 (上包絡)
    # distance 可以根據數據密度調整，這裡預設 10
    peaks, _ = find_peaks(y, distance=10)
    t_peaks = t[peaks]
    y_peaks = y[peaks]

    # 3. 繪製位移 vs 時間
    st.write("### 位移 vs 時間")
    fig1, ax1 = plt.subplots()
    ax1.plot(t, y, label='原始信號', alpha=0.6)
    ax1.scatter(t_peaks, y_peaks, color='red', s=10, label='偵測到的峰值')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Displacement")
    ax1.legend()
    st.pyplot(fig1)

    # 4. 對數回歸分析 (取 ln)
    # 注意：y 必須為正值才能取 ln
    st.write("### ln(上包絡位移) vs 時間")
    ln_y_peaks = np.log(y_peaks[y_peaks > 0]) # 確保只取正值
    t_peaks_filtered = t_peaks[y_peaks > 0]

    if len(t_peaks_filtered) > 1:
        # 執行線性回歸
        X = t_peaks_filtered.reshape(-1, 1)
        model = LinearRegression().fit(X, ln_y_peaks)
        y_pred = model.predict(X)
        r_squared = model.score(X, ln_y_peaks)

        fig2, ax2 = plt.subplots()
        ax2.scatter(t_peaks_filtered, ln_y_peaks, color='red', label='ln(Peak)')
        ax2.plot(t_peaks_filtered, y_pred, color='blue', label=f'Fit: ln(A) = {model.coef_[0]:.4f}t + {model.intercept_:.4f}')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("ln(Peak Amplitude)")
        ax2.legend()
        st.pyplot(fig2)

        # 顯示回歸結果
        st.write("### 回歸結果")
        st.write(f"**斜率 (衰減係數):** {model.coef_[0]:.6f}")
        st.write(f"**截距:** {model.intercept_:.6f}")
        st.write(f"**R-squared (相關係數):** {r_squared:.6f}")
    else:
        st.warning("偵測到的峰值不足，無法執行回歸分析。")

else:
    st.info("請上傳一個 CSV 檔案來開始分析（需包含時間與位移兩欄）。")
