import os
import pickle
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
import librosa.display
import librosa
import cv2
import scipy.ndimage
from tqdm import tqdm
import joblib
import config as cfg


def wav_plt(sig):
    from matplotlib._image import LANCZOS
    from matplotlib import pyplot as plt, ticker
    from matplotlib._image import LANCZOS
    plt.figure(figsize=(10, 5), dpi=300) # 调整为适合的大小 dpi=300
    librosa.display.waveshow(sig, sr=16000, color='g')
    # Customize x-axis ticks (for time in seconds)
    plt.savefig('./draw/wav.png',)
    plt.show()
    plt.close()


class FeatGen(Dataset):

    def __init__(self, mode=cfg.mode, train_joblib=cfg.train_joblib, test_joblib=cfg.test_joblib,data_name=cfg.data_name,
                secID=cfg.secID, cur_domain=cfg.cur_domain, year='2025', wav_dir=cfg.wav_dir):
        super(FeatGen, self).__init__()
        self.mode = mode
        self.seed = cfg.seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.wav_dir = wav_dir
        self.data_dir = cfg.data_dir
        self.feat_dir = cfg.feat_dir
        self.train_joblib = train_joblib
        self.test_joblib = test_joblib
        self.data_name = data_name
        self.secID = secID
        self.cur_domain = cur_domain
        data_choice = {'train': "train_data", 'test': "test_data"}
        self.load_dir = os.path.join(self.wav_dir, data_choice[self.mode])
        self.label_to_idx = {j:i for i,j in cfg.mach_index.items()}
        self.detect_to_idx = {'normal':0, 'anomaly':1}
        self.scaler = StandardScaler()  # 创建一个标准化器对象
        self.class_scalers_path = f'{cfg.feat_dir}/{data_name}_{year}_scaler.pkl'

    ### 原始信号零均值化和最大值归一化
    def get_normalized_audio(self, y, head_room=1e-8):
        mean_value = np.mean(y)
        y -= mean_value
        max_value = max(abs(y)) + head_room
        return y / max_value

    ### 提取logmel特征
    def compute_logmel(self, y):
        mel_spec = librosa.feature.melspectrogram(y=y, sr=cfg.fs, n_mels=cfg.n_mels, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
        logmel_spec = librosa.power_to_db(mel_spec)
        logmel_spec = logmel_spec[:, :313]
        return logmel_spec

    # 向前或向后偏移谱图 1 帧，然后与原谱图做差（绝对值差异或直接残差）。
    def compute_tsr(self, logmel, shift=1, mode='abs', pad_mode='edge'):
        """
        Temporal Shift Residual (TSR)
        Args:
            logmel (np.ndarray): 2D array of shape (freq, time)
            shift (int): number of frames to shift along time axis. Positive = right, Negative = left
            mode (str): 'diff' = signed residual, 'abs' = absolute difference
            pad_mode (str): currently only 'edge' (pad with border value)

        Returns:
            tsr (np.ndarray): residual map after temporal shift
        """
        if shift == 0:
            return np.zeros_like(logmel)

        # Time-axis shift (axis=1)
        shifted = np.roll(logmel, shift=shift, axis=1)

        if pad_mode == 'edge':
            if shift > 0:
                # First `shift` frames become invalid, pad with first frame
                shifted[:, :shift] = logmel[:, [0]]
            elif shift < 0:
                # Last `abs(shift)` frames become invalid, pad with last frame
                shifted[:, shift:] = logmel[:, [-1]]
        else:
            raise ValueError(f"Unsupported pad_mode: {pad_mode}")

        # plt.imshow(shifted, cmap='gray'), plt.show(), plt.close()
        if mode == 'diff':
            tsr = logmel - shifted
        elif mode == 'abs':
            tsr = np.abs(logmel - shifted)
        else:
            raise ValueError("mode must be 'diff' or 'abs'")

        return tsr

    # FSR（频率方向一阶差分 + 绝对值）
    def compute_fsr(self, logmel, shift=1, mode='abs', pad_mode='edge'):
        """
        Frequency Shift Residual (FSR) - 不丢频带版本
        Args:
            logmel (np.ndarray): [n_mels, T]
            shift (int): 频率轴上向上（正）或向下（负）偏移的 bin 数
            mode (str): 'diff' for signed residual, 'abs' for absolute difference
            pad_mode (str): 'edge'（默认）：边缘复制填充
        Returns:
            fsr (np.ndarray): Same shape as logmel (n_mels, T)
        """
        if shift == 0:
            return np.zeros_like(logmel)

        # 在频率轴（axis=0）上 roll
        shifted = np.roll(logmel, shift=shift, axis=0)

        if pad_mode == 'edge':
            if shift > 0:
                shifted[:shift, :] = logmel[[0], :]  # 顶部填充
            elif shift < 0:
                shifted[shift:, :] = logmel[[-1], :]  # 底部填充
        else:
            raise ValueError(f"Unsupported pad_mode: {pad_mode}")

        if mode == 'diff':
            fsr = logmel - shifted
        elif mode == 'abs':
            fsr = np.abs(logmel - shifted)
        else:
            raise ValueError("mode must be 'diff' or 'abs'")

        return fsr

    # DoG（高斯差分滤波器）
    def compute_dog(self, logmel):
        """
        DoG: Difference of Gaussian filter.
        Input: [n_mels, T] numpy array
        Output: [n_mels, T] numpy array
        """
        blur1 = cv2.GaussianBlur(logmel, (3, 3), 0.5)
        blur2 = cv2.GaussianBlur(logmel, (3, 3), 1.0)
        dog = blur1 - blur2
        return dog

    #  TKEO 信号级扰动 + logmel 提谱
    def teager_energy_signal(self, y):
        return y[1:-1] ** 2 - y[:-2] * y[2:]
    def compute_tkeo_logmel(self, y):
        y_tkeo = self.teager_energy_signal(y)
        tkeo_logmel_spec = self.compute_logmel(y_tkeo)
        return tkeo_logmel_spec

    # 多特征联合提取。
    def extract_all_features(self, audio_path):
        y, _ = librosa.load(audio_path, sr=cfg.fs)
        ### 是否零均值标准化
        if cfg.zero_normal: y = self.get_normalized_audio(y)  # 信号标准化

        # 基础谱图
        mel = librosa.feature.melspectrogram(y=y, sr=cfg.fs, n_mels=cfg.n_mels, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
        logmel = librosa.power_to_db(mel)
        logmel = logmel[:, :313]

        # TKEO
        tkeo = self.compute_tkeo_logmel(y)
        # TSR
        tsr = self.compute_tsr(tkeo)
        # FSR
        fsr = self.compute_fsr(tkeo)

        # shape
        min_t = min(logmel.shape[1], tkeo.shape[1])

        # 对齐时间维度
        logmel = logmel[:, :min_t].astype(np.float32)
        tsr = tsr[:, :min_t].astype(np.float32)
        fsr = fsr[:, :min_t].astype(np.float32)
        tkeo = tkeo[:, :min_t].astype(np.float32)

        return logmel, tsr, fsr, tkeo

    # 提取单个特征。利于特征加载和消融控制。
    def extract_single_features(self, audio_path, need_feat):
        y, _ = librosa.load(audio_path, sr=cfg.fs)
        ### 是否零均值标准化
        if cfg.zero_normal: y = self.get_normalized_audio(y)  # 信号标准化
        # shape
        min_t = 313  # min(logmel.shape[1], tkeo.shape[1])
        tkeo_logmel = self.compute_tkeo_logmel(y)
        feats = {
            'logmel': self.compute_logmel(y),
            'tsr': self.compute_tsr(tkeo_logmel),
            'fsr': self.compute_fsr(tkeo_logmel),
            'tkeo': tkeo_logmel
        }
        feat = feats[need_feat]
        # 对齐时间维度
        feat = feat[:, :min_t].astype(np.float32)

        return feat


    # 多特征生成 类别标准化
    def feat_process(self):
        allfeats = []  # 所有类别的特征信息  [(mel_spec, label, detect), ...]
        class_scalers = {str(idx_):[] for idx_ in range(cfg.class_num)}
        folder_list = os.listdir(self.load_dir)
        # 加载
        for idx, folder in enumerate(folder_list):
            folder_path = os.path.join(self.load_dir, folder)  # 文件夹路径
            label_idx = self.label_to_idx[folder]
            label_list = []
            detect_list = []
            feats = {'logmel':[],'tsr':[],'fsr':[],'tkeo':[]}

            wav_list = os.listdir(folder_path)
            inner_progress_bar = tqdm(enumerate(wav_list), total=len(wav_list), desc=f"Feat Gen: ({idx+1}/{len(folder_list)}-{folder})", position=0, leave=True)
            for subidx, filename in inner_progress_bar:
                if filename.split('_')[1] == self.secID and (filename.split('_')[2] == self.cur_domain):
                    #if subidx == 10:break
                    file_path = os.path.join(folder_path, filename)  # 文件路径
                    label_list.append(label_idx)  # 添加标签至列表
                    detect_label = filename.split('_')[4]
                    detect_label = self.detect_to_idx[detect_label]
                    detect_list.append(detect_label)  # 添加正常异常值
                    if draw_mel is True:
                        audio, _ = librosa.load(file_path, sr=None)
                        wav_plt(audio)
                        break

                    # 特征提取
                    logmel, tsr, fsr, tkeo = self.extract_all_features(file_path)
                    #plt.imshow(logmel, origin='lower', aspect='auto', cmap='jet'), plt.show(), plt.close()
                    #plt.imshow(tsr, origin='lower', aspect='auto', cmap='jet'), plt.show(), plt.close()
                    #plt.imshow(fsr, origin='lower', aspect='auto', cmap='jet'), plt.show(), plt.close()
                    #plt.imshow(tkeo, origin='lower', aspect='auto', cmap='jet'), plt.show(), plt.close()
                    #feats['wav'].append(wav)
                    feats['logmel'].append(logmel)
                    feats['tsr'].append(tsr)
                    feats['fsr'].append(fsr)
                    feats['tkeo'].append(tkeo)

            # 针对每个类别计算均值和方差
            if self.mode == 'train':  # 当训练模式时 保留计算的标准化参数
                if not os.path.exists(self.class_scalers_path):
                    for key, feat in feats.items():  # [1:]
                        feat_vstack = np.hstack(feat)  # 将所有该类别的数据堆叠
                        scaler = StandardScaler()
                        scaler.fit(feat_vstack.T)  # 计算该类别的均值和方差
                        class_scalers[str(label_idx)].append(scaler)  # 保存每个类别的标准化器
                    #scaler = class_scalers[str(label_idx)]
                else:
                    with open(self.class_scalers_path, 'rb') as f:
                        class_scalers = pickle.load(f)
                        #scaler = class_scalers[str(label_idx)]
                scaler = class_scalers[str(label_idx)]
            else:
                with open(self.class_scalers_path, 'rb') as f:
                    class_scalers = pickle.load(f)
                    scaler = class_scalers[str(label_idx)]

            ### 是否进行特征的标准化
            #if cfg.feat_normal:
            #norm_wav = feats['wav']
            norm_logmel_specs = [np.array(scaler[0].transform(i.T)).T.astype(np.float32) for i in feats['logmel']]
            norm_tsr_specs = [np.array(scaler[1].transform(i.T)).T.astype(np.float32) for i in feats['tsr']]
            norm_fsr_specs = [np.array(scaler[2].transform(i.T)).T.astype(np.float32) for i in feats['fsr']]
            norm_tkeo_specs = [np.array(scaler[3].transform(i.T)).T.astype(np.float32) for i in feats['tkeo']]

            # 集合梅尔频谱和标签
            folder_data = [((logmel, tsr, fsr, tkeo), label, detect) for logmel, tsr, fsr, tkeo, label, detect in
                           zip(norm_logmel_specs, norm_tsr_specs, norm_fsr_specs, norm_tkeo_specs, label_list, detect_list)]
            allfeats.extend(folder_data)

        # 保存类别标准化参数
        if self.mode == 'train':
            if not os.path.exists(self.class_scalers_path):
                with open(self.class_scalers_path, 'wb') as f:
                    pickle.dump(class_scalers, f)

        # 保存特征文件
        cur_joblib = {'train': self.train_joblib, 'test': self.test_joblib}
        feat_save_path = os.path.join(self.feat_dir, cur_joblib[self.mode])
        if not os.path.exists(feat_save_path):
            joblib.dump(allfeats, feat_save_path, compress=3)
        # 加载
        #my_object = joblib .load('my_object.joblib')

    # 多特征生成 全局标准化
    def feat_whole_process(self):
        label_list = []
        detect_list = []
        feats = {'logmel': [], 'tsr': [], 'fsr': [], 'tkeo': []}
        class_scalers = []
        folder_list = os.listdir(self.load_dir)
        # 加载
        for idx, folder in enumerate(folder_list):
            folder_path = os.path.join(self.load_dir, folder)  # 文件夹路径
            label_idx = self.label_to_idx[folder]
            wav_list = os.listdir(folder_path)
            inner_progress_bar = tqdm(enumerate(wav_list), total=len(wav_list), desc=f"Feat Gen: ({idx+1}/{len(folder_list)}-{folder})", position=0, leave=True)
            for subidx, filename in inner_progress_bar:
                if filename.split('_')[1] == self.secID and (filename.split('_')[2] == self.cur_domain):
                    #if subidx == 10:break
                    file_path = os.path.join(folder_path, filename)  # 文件路径
                    label_list.append(label_idx)  # 添加标签至列表
                    detect_label = filename.split('_')[4]
                    detect_label = self.detect_to_idx[detect_label]
                    detect_list.append(detect_label)  # 添加正常异常值
                    if draw_mel is True:
                        audio, _ = librosa.load(file_path, sr=None)
                        wav_plt(audio)
                        break

                    # 特征提取
                    logmel, tsr, fsr, tkeo = self.extract_all_features(file_path)
                    #plt.imshow(logmel, origin='lower', aspect='auto', cmap='jet'), plt.show(), plt.close()
                    #plt.imshow(tsr, origin='lower', aspect='auto', cmap='jet'), plt.show(), plt.close()
                    #plt.imshow(fsr, origin='lower', aspect='auto', cmap='jet'), plt.show(), plt.close()
                    #plt.imshow(tkeo, origin='lower', aspect='auto', cmap='jet'), plt.show(), plt.close()
                    #feats['wav'].append(wav)
                    feats['logmel'].append(logmel)
                    feats['tsr'].append(tsr)
                    feats['fsr'].append(fsr)
                    feats['tkeo'].append(tkeo)

        # 针对每个类别计算均值和方差
        if self.mode == 'train':  # 当训练模式时 保留计算的标准化参数
            if not os.path.exists(self.class_scalers_path):
                for key, feat in feats.items():  # [1:]
                    feat_vstack = np.hstack(feat)  # 将所有该类别的数据堆叠
                    scaler = StandardScaler()
                    scaler.fit(feat_vstack.T)  # 计算该类别的均值和方差
                    class_scalers.append(scaler)  # 保存每个类别的标准化器
            else:
                with open(self.class_scalers_path, 'rb') as f:
                    class_scalers = pickle.load(f)
            scaler = class_scalers
        else:
            with open(self.class_scalers_path, 'rb') as f:
                class_scalers = pickle.load(f)
                scaler = class_scalers

        ### 是否进行特征的标准化
        #if cfg.feat_normal:
        #norm_wav = feats['wav']
        norm_logmel_specs = [np.array(scaler[0].transform(i.T)).T.astype(np.float32) for i in feats['logmel']]
        norm_tsr_specs = [np.array(scaler[1].transform(i.T)).T.astype(np.float32) for i in feats['tsr']]
        norm_fsr_specs = [np.array(scaler[2].transform(i.T)).T.astype(np.float32) for i in feats['fsr']]
        norm_tkeo_specs = [np.array(scaler[3].transform(i.T)).T.astype(np.float32) for i in feats['tkeo']]

        # 集合梅尔频谱和标签   所有类别的特征信息  [(mel_spec, label, detect), ...]
        allfeats = [((logmel, tsr, fsr, tkeo), label, detect) for logmel, tsr, fsr, tkeo, label, detect in
                       zip(norm_logmel_specs, norm_tsr_specs, norm_fsr_specs, norm_tkeo_specs, label_list, detect_list)]

        # 保存类别标准化参数
        if self.mode == 'train':
            if not os.path.exists(self.class_scalers_path):
                with open(self.class_scalers_path, 'wb') as f:
                    pickle.dump(class_scalers, f)

        # 保存特征文件
        cur_joblib = {'train': self.train_joblib, 'test': self.test_joblib}
        feat_save_path = os.path.join(self.feat_dir, cur_joblib[self.mode])
        if not os.path.exists(feat_save_path):
            joblib.dump(allfeats, feat_save_path, compress=3)
        # 加载
        #my_object = joblib .load('my_object.joblib')


if __name__ == '__main__':
    draw_mel = False

    modes = ['train','test']  #
    #modes = ['test']  # 'train',
    train_joblibs = ['train_logmels24.joblib']  # 'train_logmelsW25.joblib'
    test_joblibs = ['test_logmels24.joblib']  # 'test_logmelsW25.joblib'
    #test_joblibs = ['test_logmelt25.joblib']  # 'test_logmeltW25.joblib'
    for mode_ in modes:
        for (train_joblib, test_joblib) in zip(train_joblibs,test_joblibs):
        #for test_joblib in test_joblibs:
            train_data = FeatGen(mode=mode_,train_joblib=train_joblib, test_joblib=test_joblib,cur_domain='source', year='2024')
            #train_data = FeatGen(mode=mode_, test_joblib=test_joblib, cur_domain='target', year='2025')
            train_data.feat_whole_process()  # whole_

    modes = ['train','test']  #
    #modes = ['test']  # 'train',
    train_joblibs = ['train_logmelts24.joblib']  # 'train_logmelsW25.joblib'
    test_joblibs = ['test_logmelts24.joblib']  # 'test_logmelsW25.joblib'
    #test_joblibs = ['test_logmelt25.joblib']  # 'test_logmeltW25.joblib'
    for mode_ in modes:
        for (train_joblib, test_joblib) in zip(train_joblibs,test_joblibs):
        #for test_joblib in test_joblibs:
            train_data = FeatGen(mode=mode_,train_joblib=train_joblib, test_joblib=test_joblib,cur_domain='target', year='2024')
            #train_data = FeatGen(mode=mode_, test_joblib=test_joblib, cur_domain='target', year='2025')
            train_data.feat_whole_process()  # whole_

    modes = ['train','test']  #
    secIDs = ['00','01','02']
    train_joblibs = ['train_logmels2200.joblib','train_logmels2201.joblib','train_logmels2202.joblib']  # 'train_logmelsW25.joblib'
    test_joblibs = ['test_logmels2200.joblib','test_logmels2201.joblib','test_logmels2202.joblib']  # 'test_logmelsW25.joblib'

    for mode_ in modes:
        for (train_joblib, test_joblib, secID) in zip(train_joblibs,test_joblibs, secIDs):
            train_data = FeatGen(mode=mode_,train_joblib=train_joblib, test_joblib=test_joblib,cur_domain='source', year='2022', secID=secID)
            train_data.feat_whole_process()  # whole_

    train_joblibs = ['train_logmelts2200.joblib','train_logmelts2201.joblib','train_logmelts2202.joblib']  # 'train_logmelsW25.joblib'
    test_joblibs = ['test_logmelts2200.joblib','test_logmelts2201.joblib','test_logmelts2202.joblib']  # 'test_logmelsW25.joblib'
    for mode_ in modes:
        for (train_joblib, test_joblib, secID) in zip(train_joblibs,test_joblibs, secIDs):
            train_data = FeatGen(mode=mode_,train_joblib=train_joblib, test_joblib=test_joblib,cur_domain='target', year='2022', secID=secID)
            train_data.feat_whole_process()  # whole_

