import numpy as np
import scipy.signal as signal
from scipy.io import loadmat
from scipy import interpolate
import os
from scipy.signal import butter, filtfilt
from tqdm import tqdm
from Feature_Extraction import *
import pickle




def process_emg_data(base_dir='D:\\DataSet', num_train=32, num_test=8, duration=2000, interp_type='quadratic'):
    X_train = np.zeros((37 * 12 * num_train, 400, 8), dtype=np.float64)  # 每类动作400帧8通道特征
    Y_train = np.zeros((37 * 12 * num_train,))
    X_test = np.zeros((37 * 12 * num_test, 400, 8), dtype=np.float64)
    Y_test = np.zeros((37 * 12 * num_test,))

    train_count = 0
    test_count = 0

    all_participants = np.arange(1, 41)
    train_participants = np.arange(1, num_train + 1)
    test_participants = np.arange(num_train + 1, 41)

    fs = 200  # 采样频率

    for b in all_participants:
        print(f"Processing participant Sub_{b}")
        for folder in range(1, 13):
            print(f"  Session {folder}")
            emg_path = os.path.join(base_dir, f'Sub_{b}', f'Session_{folder}', 'sEMG', 'emg.mat')
            label_path = os.path.join(base_dir, f'Sub_{b}', f'Session_{folder}', 'sEMG', f'{folder}EMGLabels.mat')

            if not os.path.exists(emg_path) or not os.path.exists(label_path):
                print(f"  Missing file for Sub_{b}, Session_{folder}")
                continue

            emgmat = loadmat(emg_path)
            labelmat = loadmat(label_path)

            emg_data = emgmat['emgArray'][:, 2:10]  # 第3-10列 (索引2~9)，共8通道
            labels = labelmat['Labels'].squeeze()

            for i in range(1, 38):  # 37种动作
                indices = np.where(labels == i)[0]
                emg_data_i = emg_data[indices]
                if emg_data_i.shape[0] == 0:
                    continue

                emg_signal = np.zeros((duration, 8))
                if len(emg_data_i) <= duration:
                    for j in range(8):
                        x = np.linspace(0, 1, len(emg_data_i))
                        y = emg_data_i[:, j]
                        f = interpolate.interp1d(x, y, kind=interp_type)
                        xnew = np.linspace(0, 1, duration)
                        emg_signal[:, j] = f(xnew)
                else:
                    emg_signal = emg_data_i[:duration, :]

                for k in range(8):
                    window_length = int(0.025 * fs)  # 窗长 5
                    step_size = window_length
                    num_windows = (duration - window_length) // step_size + 1
                    features = np.zeros((num_windows,))

                    for m in range(num_windows):
                        start = m * step_size
                        end = start + window_length
                        segment = emg_signal[start:end, k]
                        feature = RMS(segment)
                        features[m] = feature

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

    save_dir = 'D:/ProcessedData/sEMG/'  # 可修改为服务器路径
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'emg_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'label_train.npy'), Y_train)
    np.save(os.path.join(save_dir, 'emg_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'label_test.npy'), Y_test)

    print("处理完成：")
    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)



def Cross_Validation_process_emg_data(base_dir = 'D:\\DataSet', num_train=40, num_test=40, duration=2000, interp_type='quadratic'):
    X_train = np.zeros((37 * 10 * num_train, 400, 8), dtype=np.float64)  # 37表示37个动作，12表示单个个体12组，num_train表示个体的数目，400表示数据长度，8表示8个通道
    Y_train = np.zeros((37 * 10 * num_train,))
    X_test = np.zeros((37 * 2 * num_test, 400, 8), dtype=np.float64)
    Y_test = np.zeros((37 * 2 * num_test,))

    train_count = 0
    test_count = 0
    target_count = 0

    all_participants = np.arange(1, 41)

    # 定义滤波器参数
    fs = 200  # 采样频率
    cutoff = 10  # 截止频率
    order = 4  # 滤波器阶数

    for b in tqdm(all_participants, desc="Processing Training Data"):
        print(f"Processing participant {b}")
        for folder in range(1, 13):
            print(f"Processing folder {folder}")
            emg_path = os.path.join(base_dir, f'Sub_{b}', f'Session_{folder}', 'sEMG', 'emg.mat')
            label_path = os.path.join(base_dir, f'Sub_{b}', f'Session_{folder}', 'sEMG', f'{folder}EMGLabels.mat')
            emgmat = loadmat(emg_path)
            labelmat = loadmat(label_path)
            emg_data = emgmat['emgArray'][:, 2:10]  # 提取3-10列，即8通道数据
            labels = labelmat['Labels']

            for i in range(1, 38):
                indices = np.where(labels == i)[0]
                emg_data_i = emg_data[indices]
                emg_signal = np.zeros((duration, 8))

                if len(emg_data_i) <= duration:
                    for j in range(0, 8):
                        x = np.linspace(0, 1, len(emg_data_i))
                        y = emg_data_i[:, j]
                        f = interpolate.interp1d(x, y, kind=interp_type)
                        xnew = np.linspace(0, 1, duration)
                        emg_signal[:, j] = f(xnew)
                else:
                    emg_signal = emg_data_i[0:duration, :]

                for k in range(8):
                    window_length = int(0.025 * 200)  # 滑动窗为5
                    step_size = window_length  # 滑动步长为5
                    num_windows = (duration - window_length) // step_size + 1  # 窗口的个数
                    features = np.zeros((num_windows,))  # 初始化feature的格式为(5, ),每5个sEMG信号特征(一个窗口)产生一个feature

                    for m in range(num_windows):
                        start = m * step_size
                        end = start + window_length
                        segment = emg_signal[start:end, k]
                        feature = RMS(segment)
                        features[m] = feature

                    if folder in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:  # 后2个文件作为测试集
                        X_train[train_count, :, k] = features
                    else:
                        X_test[test_count, :, k] = features

                if folder in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                    Y_train[train_count] = i - 1
                    train_count += 1
                else:
                    Y_test[test_count] = i - 1
                    test_count += 1

    save_dir = 'D:/ProcessedData/sEMG/'
    np.save(os.path.join(save_dir, 'emg_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'label_train.npy'), Y_train)
    np.save(os.path.join(save_dir, 'emg_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'label_test.npy'), Y_test)

    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)




if __name__ == "__main__":
    # process_emg_data()
    Cross_Validation_process_emg_data()

