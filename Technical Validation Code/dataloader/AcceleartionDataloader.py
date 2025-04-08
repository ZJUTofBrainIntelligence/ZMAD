import numpy as np
import scipy.signal as signal
from scipy.io import loadmat
from scipy import interpolate
import os
from tqdm import tqdm
from Feature_Extraction import *
import matplotlib.pyplot as plt

import numpy as np
import os
from scipy.io import loadmat
from scipy import interpolate
from tqdm import tqdm

def RMS(signal):
    return np.sqrt(np.mean(signal ** 2))

def Cross_Validation_process_Acc_data(base_dir='D:\\DataSet', num_train=40, num_test=40, duration=248, interp_type='quadratic'):
    X_train = np.zeros((37 * 10 * num_train, duration, 3), dtype=np.float64)  # 3通道加速度数据
    Y_train = np.zeros((37 * 10 * num_train,))
    X_test = np.zeros((37 * 2 * num_test, duration, 3), dtype=np.float64)
    Y_test = np.zeros((37 * 2 * num_test,))

    fs = 200  # 采样频率
    window_length = int(0.025 * fs)  # 5帧窗口
    step_size = window_length // 2
    num_windows = (duration - window_length) // step_size + 1

    train_count = 0
    test_count = 0

    for subj in tqdm(range(1, 41), desc='Processing Participants'):
        subj_dir = os.path.join(base_dir, f'Sub_{subj}')
        for session in range(1, 13):
            session_dir = os.path.join(subj_dir, f'Session_{session}')
            acc_path = os.path.join(session_dir, 'Acceleration', 'acceleration.mat')
            label_path = os.path.join(session_dir, 'Acceleration', f'{session}AccLabels.mat')

            if not os.path.exists(acc_path) or not os.path.exists(label_path):
                print(f"Missing data at: {acc_path} or {label_path}")
                continue

            acc_data = loadmat(acc_path)['Acceleration'][:, 2:5]  # 第3-5列，3通道
            labels = loadmat(label_path)['Labels'].flatten()

            for label in range(1, 38):
                indices = np.where(labels == label)[0]
                if len(indices) == 0:
                    continue

                data_slice = acc_data[indices]
                signal = np.zeros((duration, 3))
                for ch in range(3):
                    x = np.linspace(0, 1, len(data_slice))
                    y = data_slice[:, ch]
                    f = interpolate.interp1d(x, y, kind=interp_type, fill_value="extrapolate")
                    x_new = np.linspace(0, 1, duration)
                    signal[:, ch] = f(x_new)

                features = np.zeros((duration, 3))
                for ch in range(3):
                    feat = np.zeros((num_windows,))
                    for m in range(num_windows):
                        start = m * step_size
                        end = start + window_length
                        segment = signal[start:end, ch]
                        feat[m] = RMS(segment)
                    features[:num_windows, ch] = feat

                if session <= 10:
                    X_train[train_count, :, :] = features
                    Y_train[train_count] = label - 1
                    train_count += 1
                else:
                    X_test[test_count, :, :] = features
                    Y_test[test_count] = label - 1
                    test_count += 1

    save_dir = 'D:\\Processed\\Acceleration'
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'Acceleration_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'Acceleration_label_train.npy'), Y_train)
    np.save(os.path.join(save_dir, 'Acceleration_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'Acceleration_label_test.npy'), Y_test)

    print("Finished processing.")
    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)



def FFT(segment, fs):
    freqs = np.fft.rfftfreq(len(segment), 1/fs)
    psd = np.abs(np.fft.rfft(segment)) ** 2
    return freqs, psd

def mean_frequency(psd, freqs):
    return np.sum(psd * freqs) / np.sum(psd)

def process_cross_subject_acc_data(base_dir='D:/DataSet', num_train=32, num_test=8, duration=500, interp_type='quadratic'):
    X_train = np.zeros((37 * 12 * num_train, 248, 3), dtype=np.float64)
    Y_train = np.zeros((37 * 12 * num_train,))
    X_test = np.zeros((37 * 12 * num_test, 248, 3), dtype=np.float64)
    Y_test = np.zeros((37 * 12 * num_test,))

    train_count, test_count = 0, 0
    train_ids = np.arange(1, 33)
    test_ids = np.arange(33, 41)

    fs = 200  # 采样频率

    for subj in range(1, 41):
        print(f"Processing Subject {subj}")
        subj_folder = f"Sub_{subj}"

        for sess in range(1, 13):
            sess_folder = f"Session_{sess}"
            acc_path = os.path.join(base_dir, f"Sub_{subj}", f"Session_{sess}", 'Acceleration', 'acceleration.mat')
            label_path = os.path.join(base_dir, f"Sub_{subj}", f"Session_{sess}", 'Acceleration', f'{sess}AccLabels.mat')

            print(acc_path, label_path)

            try:
                acc_mat = loadmat(acc_path)
                label_mat = loadmat(label_path)
            except FileNotFoundError:
                print(f"Missing file at subject {subj}, session {sess}")
                continue

            acc_data = acc_mat['Acceleration'][:, 2:5]  # 第3-5列
            labels = label_mat['Labels'].flatten()

            for action in range(1, 38):
                indices = np.where(labels == action)[0]
                acc_action = acc_data[indices]

                if acc_action.size == 0:
                    continue

                if len(acc_action) < duration:
                    acc_interp = np.zeros((duration, 3))
                    for ch in range(3):
                        x = np.linspace(0, 1, len(acc_action))
                        y = acc_action[:, ch]
                        f = interpolate.interp1d(x, y, kind=interp_type, fill_value="extrapolate")
                        acc_interp[:, ch] = f(np.linspace(0, 1, duration))
                else:
                    acc_interp = acc_action[:duration]

                for ch in range(3):
                    win_len = int(0.025 * fs)
                    step = win_len // 2
                    num_win = (duration - win_len) // step + 1
                    feats = np.zeros(num_win)

                    for m in range(num_win):
                        seg = acc_interp[m * step : m * step + win_len, ch]
                        freqs, psd = FFT(seg, fs)
                        feats[m] = mean_frequency(psd, freqs)

                    if subj in train_ids:
                        X_train[train_count, :, ch] = feats
                    elif subj in test_ids:
                        X_test[test_count, :, ch] = feats

                if subj in train_ids:
                    Y_train[train_count] = action - 1
                    train_count += 1
                elif subj in test_ids:
                    Y_test[test_count] = action - 1
                    test_count += 1

    save_dir = 'D:/'
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'Acceleration_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'Acceleration_label_train.npy'), Y_train)
    np.save(os.path.join(save_dir, 'Acceleration_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'Acceleration_label_test.npy'), Y_test)

    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)

if __name__ == "__main__":
    # Cross_Validation_process_Acc_data()
    process_cross_subject_acc_data()
