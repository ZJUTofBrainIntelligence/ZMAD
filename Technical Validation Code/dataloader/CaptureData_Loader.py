import numpy as np
import scipy.signal as signal
from scipy.io import loadmat
from scipy import interpolate
import os
from tqdm import tqdm
from Feature_Extraction import *
import matplotlib.pyplot as plt

def process_capture_data(base_dir='D:\\DataSet', num_train=32, num_test=8, duration=600, interp_type='quadratic'):
    X_train = np.zeros((37 * 12 * num_train, 298, 3), dtype=np.float64)
    Y_train = np.zeros((37 * 12 * num_train,))
    X_test = np.zeros((37 * 12 * num_test, 298, 3), dtype=np.float64)
    Y_test = np.zeros((37 * 12 * num_test,))

    train_count = 0
    test_count = 0

    all_participants = np.arange(1, 41)
    train_participants = np.arange(1, 33)
    test_participants = np.arange(33, 41)

    fs = 200  # 采样频率

    for b in all_participants:
        print(f"Processing participant {b}")
        for folder in range(1, 13):
            print(f"  Session {folder}")
            # 构造路径
            capture_path = os.path.join(base_dir, f'Sub_{b}', f'Session_{folder}', 'Optitrack', 'RigidBody1.mat')
            label_path = os.path.join(base_dir, f'Sub_{b}', f'Session_{folder}', 'Optitrack', f'{folder}TrajectoryLabels.mat')

            # 检查文件是否存在
            if not os.path.exists(capture_path) or not os.path.exists(label_path):
                print(f"  Missing file at subject {b}, session {folder}")
                continue

            capturemat = loadmat(capture_path)
            labelmat = loadmat(label_path)

            capture_data = capturemat['data'][:, 2:5]  # 第3~5列
            labels = labelmat['Labels'].squeeze()

            for i in range(1, 38):  # 动作类别 1~37
                indices = np.where(labels == i)[0]
                capture_data_i = capture_data[indices]

                if capture_data_i.size == 0:
                    continue

                # 插值或裁剪
                if len(capture_data_i) <= duration:
                    capture_signal = np.zeros((duration, 3))
                    for j in range(3):
                        x = np.linspace(0, 1, len(capture_data_i))
                        y = capture_data_i[:, j]
                        f = interpolate.interp1d(x, y, kind=interp_type)
                        xnew = np.linspace(0, 1, duration)
                        capture_signal[:, j] = f(xnew)
                else:
                    capture_signal = capture_data_i[:duration, :]

                for k in range(3):
                    window_length = int(0.025 * fs)  # 5点窗
                    step_size = window_length // 2
                    num_windows = (duration - window_length) // step_size + 1
                    features = np.zeros((num_windows,))

                    for m in range(num_windows):
                        start = m * step_size
                        end = start + window_length
                        segment = capture_signal[start:end, k]
                        features[m] = RMS(segment)

                    if b in train_participants:
                        X_train[train_count, :, k] = features
                    elif b in test_participants:
                        X_test[test_count, :, k] = features

                if b in train_participants:
                    Y_train[train_count] = i - 1
                    train_count += 1
                elif b in test_participants:
                    Y_test[test_count] = i - 1
                    test_count += 1

    save_dir = 'D:/ProcessedData/CaptureData/'  # 请根据需要修改保存路径
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'CaptureData_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'CaptureData_label_train.npy'), Y_train)
    np.save(os.path.join(save_dir, 'CaptureData_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'CaptureData_label_test.npy'), Y_test)

    print("处理完成：")
    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)



if __name__ == "__main__":
    process_capture_data()