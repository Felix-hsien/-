import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, find_peaks
from scipy.interpolate import interp1d

st.title("上包絡偵測實作")

# 模擬信號
fs = 1000
t = np.linspace(0, 1, fs)
signal = (1 + 0.5 * np.sin(2 * np.pi * 5 * t)) * np.sin(2 * np.pi * 60 * t)

method = st.radio("選擇偵測方法：", ("希爾伯特轉換", "峰值插值法"))

fig, ax = plt.subplots()
ax.plot(t, signal, alpha=0.5, label='原始信號')

if method == "希爾伯特轉換":
    env = np.abs(hilbert(signal))
    ax.plot(t, env, color='red', label='Hilbert Envelope')
else:
    peaks, _ = find_peaks(signal)
    f = interp1d(t[peaks], signal[peaks], kind='cubic', fill_value="extrapolate")
    ax.plot(t, f(t), color='green', label='Interpolation Envelope')

ax.legend()
st.pyplot(fig)
