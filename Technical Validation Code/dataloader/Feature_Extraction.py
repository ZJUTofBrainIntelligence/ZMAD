
import numpy as np
import scipy
import scipy.signal as signal
from scipy.io import loadmat
from scipy import interpolate
import os
from scipy.signal import butter, filtfilt
import pywt




# 波形长度（WL）
def WL(data):
    return np.sum(np.abs(np.diff(data)))

# 斜率符号变化（SSC）
def SSC(data):

    # 计算标准差
    std_dev = np.std(data)
    # 根据标准差和倍数计算阈值
    threshold = std_dev * 2
    diff = np.diff(data)
    count = 0
    for i in range(1, len(diff)):
        if ((diff[i-1] * diff[i] < 0) and (abs(diff[i-1] - diff[i]) > threshold)):
            count += 1
    return count

def Entropy(data):
    p_data = np.histogram(data, bins=100, density=True)[0]
    epsilon = 1e-8  # 防止对零取对数
    return -np.sum(p_data * np.log(p_data + epsilon))

def Median(data):
    return np.median(data)

def iEMG(data):
    return np.sum(np.abs(data))

def ZC(data, threshold=0.01):
    data = data - np.mean(data)  # 去中心化
    crossings = np.where(np.diff(np.signbit(data)))[0]
    return len([x for x in crossings if abs(data[x]) > threshold])

def Mean(data):
    return np.mean(data)


# 峭度是度量数据分布尾部厚度的统计量
def kurtosis(data):
    n = len(data)
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:  # 避免除以零
        return 0
    kurt = ((n * (n + 1) * ((data - mean) ** 4).sum() / (n - 1) / (n - 2) / (n - 3) / (std ** 4))
            - (3 * (n - 1) ** 2 / (n - 2) / (n - 3)))
    return kurt




def MAV(data):
    result = abs(data).sum() / data.shape[0]
    return result

def Energy(data):
    result = np.square(data).sum() / data.shape[0]
    return result

def Variance(data):
    mu = np.mean(data)
    result = ((data - mu) ** 2).mean()
    return result
def RMS(data):
    result = np.sqrt(np.square(data).sum() / data.shape[0])
    return result

def ATME3(data):
    result = (data ** 3).sum() / data.shape[0]
    return result
def ATME4(data):
    result = (data ** 4).sum() / data.shape[0]
    return result
def ATME5(data):
    result = (data ** 5).sum() / data.shape[0]
    return result
def log_detector(data):
    epsilon = 1e-8  # 小的正值偏移，防止对零取对数
    result = np.exp(np.mean(np.log(np.abs(data) + epsilon)))
    return result

def STFT(data, fs=1.0, window='hann', nperseg=256, noverlap=128):
    f, t, Zxx = signal.stft(data, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
    return f, t, np.abs(Zxx)
def CWT(data, scales, wavelet_name='morl'):
    coefficients, frequencies = pywt.cwt(data, scales, wavelet_name)
    return frequencies, coefficients


def WAMP(data, multiplier=2.0):
    """
    计算威利森幅度（WAMP）。
    参数:
    data : numpy.ndarray
        输入的一维数据数组。
    multiplier : float
        标准差的倍数，用来设置动态阈值。
    返回:
    wamp : int
        威利森幅度计数，即信号幅度变化超过阈值的次数。
    """
    # 计算标准差
    std_dev = np.std(data)
    # 根据标准差和倍数计算阈值
    threshold = std_dev * multiplier
    # 计算相邻样本之间的差的绝对值
    abs_diff = np.abs(np.diff(data))
    # 计算超过阈值的次数
    wamp = np.sum(abs_diff > threshold)
    return wamp



# 设计高通巴特沃斯滤波器
def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyq  # 归一化截止频率
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

# 使用滤波器过滤数据
def butter_highpass_filter(data, cutoff, fs, order=4):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y



def FFT(window, fs):
    """Perform FFT on a given signal window and calculate Power Spectral Density (PSD)."""
    freqs = np.fft.rfftfreq(len(window), d=1/fs)
    fft_values = np.fft.rfft(window)
    psd = np.abs(fft_values) ** 2
    return freqs, psd

def mean_power(psd):
    """Calculate Mean Power (MNP) from PSD."""
    return np.mean(psd)

def mean_frequency(psd, freqs):
    """Calculate Mean Frequency (MNF) using PSD and frequencies."""
    total_power = np.sum(psd)
    return np.sum(freqs * psd) / total_power if total_power > 0 else 0

def median_frequency(psd, freqs):
    """Calculate Median Frequency (MDF) from PSD."""
    cumulative_power = np.cumsum(psd)
    median_idx = np.searchsorted(cumulative_power, cumulative_power[-1] / 2)
    return freqs[median_idx]

def peak_frequency(psd, freqs):
    """Identify Peak Frequency (PKF) from PSD."""
    peak_idx = np.argmax(psd)
    return freqs[peak_idx]

def frequency_ratio(psd, freqs, low_freq, high_freq):
    """Calculate Frequency Ratio (FR) within a specified frequency band."""
    band_power = np.sum(psd[(freqs >= low_freq) & (freqs <= high_freq)])
    total_power = np.sum(psd)
    return band_power / total_power if total_power > 0 else 0
