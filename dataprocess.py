import mne
import numpy as np
import pandas as pd
from pathlib import Path
import gc
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import warnings

import matplotlib.pyplot as plt
import matplotlib
import re
import numpy as np
from pathlib import Path
import numpy as np
import mne
import os
import re
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
import matplotlib.pyplot as plt
import pickle
import random
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')


def extract_patient_22channels_ultra_low_memory(patient_path, standard_22, 
                                                chunk_size=30,  # 减小到30秒
                                                save_dir="extracted_data"):
    """
    超低内存占用的22通道提取方案
    """
    patient_path = Path(patient_path)
    patient_id = patient_path.name
    
    print(f"\n{'='*60}")
    print(f"处理患者: {patient_id}")
    print(f"{'='*60}")
    
    # 创建保存目录
    save_path = Path(save_dir) / patient_id
    save_path.mkdir(parents=True, exist_ok=True)
    
    edf_files = sorted(patient_path.glob("*.edf"))
    print(f"找到 {len(edf_files)} 个EDF文件")
    
    successful_files = 0
    
    for edf_idx, edf_file in enumerate(edf_files, 1):
        print(f"\n{'─'*50}")
        print(f"[{edf_idx}/{len(edf_files)}] 处理: {edf_file.name}")
        
        try:
            # === 关键改进1：不加载任何数据，只读取头信息 ===
            raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)
            
            # 检查可用通道
            available_channels = []
            channel_mapping = {}
            
            for std_ch in standard_22:
                if std_ch in raw.ch_names:
                    available_channels.append(std_ch)
                    channel_mapping[std_ch] = std_ch
                elif f"{std_ch}-0" in raw.ch_names:
                    available_channels.append(f"{std_ch}-0")
                    channel_mapping[std_ch] = f"{std_ch}-0"
                    print(f"  使用 {std_ch}-0 替代 {std_ch}")
            
            if len(available_channels) != 22:
                print(f"  警告: 只有 {len(available_channels)} 个通道可用")
                continue
            
            # 获取基本信息
            sfreq = raw.info['sfreq']
            n_samples = int(raw.n_times)
            duration = n_samples / sfreq
            
            print(f"  采样率: {sfreq} Hz")
            print(f"  总时长: {duration:.1f}秒 ({duration/3600:.2f}小时)")
            
            # 创建picks
            picks = [raw.ch_names.index(ch) for ch in available_channels]
            
            # === 关键改进2：极小块大小 ===
            chunk_samples = int(chunk_size * sfreq)  # 30秒的样本数
            n_chunks = (n_samples + chunk_samples - 1) // chunk_samples
            
            print(f"  分块大小: {chunk_size}秒 ({chunk_samples}样本)")
            print(f"  分块数量: {n_chunks}")
            
            # 创建输出文件
            output_file = save_path / f"{edf_file.stem}_22ch.npz"
            
            # === 关键改进3：直接保存到磁盘，不累积在内存 ===
            all_chunks = []  # 只保存元数据，不保存数据
            
            for chunk_idx in range(n_chunks):
                try:
                    start = chunk_idx * chunk_samples
                    end = min((chunk_idx + 1) * chunk_samples, n_samples)
                    
                    # 读取当前块
                    chunk_data, chunk_times = raw[:, start:end]
                    
                    # 选择需要的通道
                    chunk_data_selected = chunk_data[picks, :]
                    
                    # 重新排列通道
                    reordered_chunk = np.zeros((22, chunk_data_selected.shape[1]), 
                                              dtype=np.float32)  # 使用float32节省内存
                    
                    for j, std_ch in enumerate(standard_22):
                        actual_ch = channel_mapping[std_ch]
                        idx = available_channels.index(actual_ch)
                        reordered_chunk[j, :] = chunk_data_selected[idx, :].astype(np.float32)
                    
                    # === 关键改进4：每个块单独保存 ===
                    chunk_file = save_path / f"{edf_file.stem}_chunk{chunk_idx:04d}.npy"
                    np.save(chunk_file, reordered_chunk)
                    
                    # 记录元数据
                    all_chunks.append({
                        'chunk_id': chunk_idx,
                        'file': str(chunk_file),
                        'start_sample': start,
                        'end_sample': end,
                        'start_time': start / sfreq,
                        'end_time': end / sfreq,
                        'n_samples': end - start
                    })
                    
                    # 显示进度
                    if (chunk_idx + 1) % 20 == 0 or chunk_idx == n_chunks - 1:
                        print(f"    进度: {chunk_idx+1}/{n_chunks} 块 "
                              f"({(chunk_idx+1)/n_chunks*100:.1f}%)")
                    
                    # 立即释放内存
                    del chunk_data, chunk_data_selected, reordered_chunk
                    
                    # 每10块强制垃圾回收
                    if (chunk_idx + 1) % 10 == 0:
                        gc.collect()
                        
                except Exception as e:
                    print(f"    块 {chunk_idx} 错误: {e}")
                    continue
            
            # === 关键改进5：保存元数据文件 ===
            if all_chunks:
                metadata = {
                    'file_name': edf_file.name,
                    'sfreq': sfreq,
                    'duration': duration,
                    'n_samples': n_samples,
                    'channels': standard_22,
                    'n_chunks': len(all_chunks),
                    'chunk_size': chunk_samples,
                    'chunks': all_chunks
                }
                
                # 保存元数据
                meta_file = save_path / f"{edf_file.stem}_metadata.json"
                import json
                with open(meta_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                successful_files += 1
                print(f"  ✅ 文件处理成功: {len(all_chunks)} 块")
            
            # 清理
            del raw
            gc.collect()
            
            # 添加延时，让系统恢复
            time.sleep(1)
            
        except Exception as e:
            print(f"  ❌ 文件错误: {e}")
            continue
    
    print(f"\n✅ 患者 {patient_id} 处理完成: {successful_files}/{len(edf_files)} 个文件")
    return successful_files

def extract_all_patients_22channels(data_path, standard_22, valid_patients):
    """
    提取所有患者的22通道数据（修复版）
    """
    print("\n" + "=" * 70)
    print("开始提取所有患者的22通道数据")
    print("=" * 70)
    print(f"数据路径: {data_path}")
    print(f"患者列表: {valid_patients}")
    print("=" * 70)
    
    all_patients_data = {}
    extraction_summary = []
    
    for patient in valid_patients:
        patient_path = Path(data_path) / patient
        
        if not patient_path.exists():
            print(f"\n⚠️ 跳过不存在的患者: {patient}")
            continue
        
        print(f"\n{'='*60}")
        print(f"处理患者: {patient}")
        print(f"{'='*60}")
        
        # 提取该患者数据
        patient_data = extract_patient_22channels_ultra_low_memory(
            patient_path, standard_22
        )
        
    
def reconstruct_patient_data(patient_id, save_dir="extracted_data"):
    """
    重建完整的患者数据
    """
    print(f"\n{'='*60}")
    print(f"重建患者 {patient_id} 数据")
    print(f"{'='*60}")
    
    patient_path = Path(save_dir) / patient_id
    
    # 找到所有元数据文件
    meta_files = list(patient_path.glob("*_metadata.json"))
    
    for meta_file in meta_files:
        print(f"\n处理: {meta_file.name}")
        
        import json
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
        
        # 获取所有块文件
        chunk_files = sorted(patient_path.glob(f"{metadata['file_name'].replace('.edf', '')}_chunk*.npy"))
        
        if len(chunk_files) != metadata['n_chunks']:
            print(f"  警告: 期望 {metadata['n_chunks']} 块, 找到 {len(chunk_files)} 块")
            continue
        
        # 计算总样本数
        total_samples = sum(chunk['n_samples'] for chunk in metadata['chunks'])
        
        # 重建完整数据
        print(f"  重建数据中...")
        full_data = np.zeros((22, total_samples), dtype=np.float32)
        
        current_pos = 0
        for chunk_info, chunk_file in zip(metadata['chunks'], chunk_files):
            chunk_data = np.load(chunk_file)
            n_samples = chunk_data.shape[1]
            full_data[:, current_pos:current_pos + n_samples] = chunk_data
            current_pos += n_samples
        
        # 创建时间数组
        times = np.arange(total_samples) / metadata['sfreq']
        
        # 保存完整文件
        output_file = patient_path / f"{metadata['file_name'].replace('.edf', '')}_full.npz"
        np.savez_compressed(
            output_file,
            data=full_data,
            times=times,
            channels=metadata['channels'],
            sfreq=metadata['sfreq'],
            duration=metadata['duration']
        )
        
        print(f"  ✅ 完整数据已保存: {output_file}")
        print(f"     数据形状: {full_data.shape}")
        
        #可选：删除块文件以释放空间
        for chunk_file in chunk_files:
            chunk_file.unlink()


class EEG2STFTConverter:
    """
    将EEG信号转换为STFT时频图
    支持不同窗口长度（5秒、15秒、30秒）
    """
    
    def __init__(self, npz_file, sfreq=256, window_lengths=[5, 15, 30], 
                 nperseg=256, noverlap=128, output_dir='stft_data', seizure_times=None,
                 global_preictal_duration=None, global_interictal_duration=None, valid_ranges=None):
        """
        参数:
            npz_file: 输入的.npz文件路径
            sfreq: 采样率 (Hz)
            window_lengths: 滑动窗口长度列表（秒）
            nperseg: STFT窗口长度（采样点）
            noverlap: STFT窗口重叠
            output_dir: 输出目录
            seizure_times: 发作时间列表，每个元素为(start, end)秒
            global_preictal_duration: 全局发作前期总时长
            global_interictal_duration: 全局发作间期总时长
            valid_ranges: 预计算的有效时间段列表
        """
        self.npz_file = npz_file
        self.sfreq = sfreq
        self.window_lengths = window_lengths
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seizure_times = seizure_times
        self.global_preictal_duration = global_preictal_duration
        self.global_interictal_duration = global_interictal_duration
        
        # 加载数据
        self._load_data()
        
        # 设置有效时间段
        if valid_ranges:
            self.valid_time_ranges = valid_ranges
            print(f"  使用预计算的有效时间段: {len(valid_ranges)}个范围")
            for i, (range_type, start, end) in enumerate(valid_ranges):
                print(f"    {i+1}. {range_type}: {start:.1f} - {end:.1f}秒")
        elif seizure_times:
            self.valid_time_ranges = self._compute_valid_time_ranges()
        else:
            self.valid_time_ranges = None
        
    def _load_data(self):
        """加载.npz文件"""
        print(f"\n{'='*60}")
        print(f"加载数据: {self.npz_file}")
        print(f"{'='*60}")
        
        data = np.load(self.npz_file)
        self.eeg_data = data['data']  # (22, n_samples)
        self.times = data['times']
        self.channels = data['channels']
        self.sfreq = data.get('sfreq', self.sfreq)
        
        print(f"数据形状: {self.eeg_data.shape}")
        print(f"通道数: {len(self.channels)}")
        print(f"采样率: {self.sfreq} Hz")
        print(f"总时长: {self.times[-1]:.1f}秒 ({self.times[-1]/3600:.2f}小时)")
    
    def _compute_valid_time_ranges(self):
        """
        计算有效时间段：
        1. 发作前期：发作前35分钟范围内
        2. 发作间期：发作后30分钟至下一次发作开始前30分钟
        3. 删除SPH（发作前5分钟）
        """
        valid_ranges = []
        total_duration = self.times[-1]
        
        # 按时间排序发作时间
        sorted_seizures = sorted(self.seizure_times, key=lambda x: x[0])
        
        print(f"\n{'='*60}")
        print("计算有效时间段")
        print(f"{'='*60}")
        print(f"总时长: {total_duration:.1f}秒")
        print(f"发作次数: {len(sorted_seizures)}")
        
        # 处理第一个发作
        first_seizure_start = sorted_seizures[0][0]
        
        # 第一个发作前的发作前期（如果有）
        pre_seizure_start = max(0, first_seizure_start - 35 * 60)  # 发作前35分钟
        sph_start = first_seizure_start - 5 * 60  # SPH开始时间
        
        if pre_seizure_start < sph_start:
            valid_ranges.append((pre_seizure_start, sph_start))
            print(f"  发作前期1: {pre_seizure_start:.1f} - {sph_start:.1f}秒")
        
        # 处理发作间期
        for i in range(len(sorted_seizures) - 1):
            current_seizure_end = sorted_seizures[i][1]
            next_seizure_start = sorted_seizures[i+1][0]
            
            # 发作间期：当前发作结束后30分钟至下一次发作开始前30分钟
            interictal_start = current_seizure_end + 30 * 60
            interictal_end = next_seizure_start - 30 * 60
            
            if interictal_start < interictal_end:
                valid_ranges.append((interictal_start, interictal_end))
                print(f"  发作间期{i+1}: {interictal_start:.1f} - {interictal_end:.1f}秒")
            
            # 下一次发作的发作前期（排除SPH）
            pre_seizure_start = max(interictal_end, next_seizure_start - 35 * 60)
            sph_start = next_seizure_start - 5 * 60
            
            if pre_seizure_start < sph_start:
                valid_ranges.append((pre_seizure_start, sph_start))
                print(f"  发作前期{i+2}: {pre_seizure_start:.1f} - {sph_start:.1f}秒")
        
        # 处理最后一个发作后的发作间期
        last_seizure_end = sorted_seizures[-1][1]
        interictal_start = last_seizure_end + 30 * 60
        if interictal_start < total_duration:
            valid_ranges.append((interictal_start, total_duration))
            print(f"  发作间期{len(sorted_seizures)}: {interictal_start:.1f} - {total_duration:.1f}秒")
        
        print(f"  有效时间段数: {len(valid_ranges)}")
        return valid_ranges
        
    def compute_stft(self, signal_data):
        """
        计算单个信号的STFT
        
        返回:
            stft_matrix: 形状 (freq_bins, time_bins)
            frequencies: 频率轴
            times: 时间轴
        """
        frequencies, times, stft_matrix = signal.spectrogram(
            signal_data,
            fs=self.sfreq,
            window='hann',
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            scaling='spectrum'
        )
        
        # 转换为功率谱 (dB)
        stft_db = 10 * np.log10(np.abs(stft_matrix) + 1e-10)
        
        return stft_db, frequencies, times
    
    def extract_windows(self, window_sec):
        """
        提取指定长度的滑动窗口
        
        参数:
            window_sec: 窗口长度（秒）
            
        返回:
            windows: 形状 (n_windows, n_channels, n_samples)
            window_times: 每个窗口的开始时间
        """
        window_samples = int(window_sec * self.sfreq)
        base_stride_samples = window_samples // 2  # 基础步长（50%重叠）
        
        # 计算调整因子
        preictal_stride_samples = base_stride_samples
        
        if self.valid_time_ranges and self.global_preictal_duration is not None and self.global_interictal_duration is not None:
            # 使用全局发作前期和发作间期总时长计算调整因子
            if self.global_interictal_duration > 0:
                adjustment_factor = max(self.global_preictal_duration / self.global_interictal_duration,0.25)
                preictal_stride_samples = max(1, int(base_stride_samples * adjustment_factor))
                print(f"  基础步长: {base_stride_samples}样本 ({base_stride_samples/self.sfreq:.1f}秒)")
                print(f"  发作前期步长: {preictal_stride_samples}样本 ({preictal_stride_samples/self.sfreq:.1f}秒)")
                print(f"  全局发作前期总时长: {self.global_preictal_duration:.1f}秒")
                print(f"  全局发作间期总时长: {self.global_interictal_duration:.1f}秒")
                print(f"  调整因子: {adjustment_factor:.2f}")
        elif self.valid_time_ranges:
            # 后备方案：使用当前文件的时长计算
            preictal_duration = 0
            interictal_duration = 0
            
            for i, (start_range, end_range) in enumerate(self.valid_time_ranges):
                duration = end_range - start_range
                # 偶数索引为发作前期，奇数索引为发作间期（根据compute_valid_time_ranges的输出顺序）
                if i % 2 == 0:
                    preictal_duration += duration
                else:
                    interictal_duration += duration
            
            if interictal_duration > 0:
                adjustment_factor = preictal_duration / interictal_duration
                preictal_stride_samples = max(1, int(base_stride_samples * adjustment_factor))
                print(f"  基础步长: {base_stride_samples}样本 ({base_stride_samples/self.sfreq:.1f}秒)")
                print(f"  发作前期步长: {preictal_stride_samples}样本 ({preictal_stride_samples/self.sfreq:.1f}秒)")
                print(f"  当前文件发作前期总时长: {preictal_duration:.1f}秒")
                print(f"  当前文件发作间期总时长: {interictal_duration:.1f}秒")
                print(f"  调整因子: {adjustment_factor:.2f}")
        
        n_samples = self.eeg_data.shape[1]
        windows = []
        window_times = []
        
        # 遍历所有可能的窗口位置
        start = 0
        while start <= n_samples - window_samples:
            window_start_time = start / self.sfreq
            window_end_time = (start + window_samples) / self.sfreq
            
            # 检查窗口是否在有效时间段内，并确定使用的步长
            use_preictal_stride = False
            if self.valid_time_ranges:
                window_valid = False
                for i, range_info in enumerate(self.valid_time_ranges):
                    # 检查是否包含类型信息
                    if len(range_info) == 3:
                        # 格式: (type, start, end)
                        range_type, start_range, end_range = range_info
                    else:
                        # 旧格式: (start, end)
                        start_range, end_range = range_info
                        range_type = 'preictal' if i % 2 == 0 else 'interictal'
                    
                    if window_start_time >= start_range and window_end_time <= end_range:
                        window_valid = True
                        use_preictal_stride = (range_type == 'preictal')
                        break
                if not window_valid:
                    # 移动到下一个位置
                    start += base_stride_samples
                    continue
            
            # 提取窗口
            end = start + window_samples
            window = self.eeg_data[:, start:end]  # (22, window_samples)
            windows.append(window)
            window_times.append(window_start_time)
            
            if use_preictal_stride:
                print(f"  窗口 {len(windows)}: 发作前期，步长 {preictal_stride_samples}样本 ({preictal_stride_samples/self.sfreq:.1f}秒)")
            else:
                print(f"  窗口 {len(windows)}: 发作间期，步长 {base_stride_samples}样本 ({base_stride_samples/self.sfreq:.1f}秒)")
                

            # 根据窗口类型选择步长
            if use_preictal_stride:
                start += preictal_stride_samples
            else:
                start += base_stride_samples
        
        windows = np.array(windows)  # (n_windows, 22, window_samples)
        print(f"  窗口长度 {window_sec}秒: {len(windows)} 个窗口")
        
        return windows, np.array(window_times)
    
    def convert_to_stft(self, window_sec):
        """
        将指定窗口长度的所有窗口转换为STFT
        
        返回:
            stft_data: 形状 (n_windows, n_channels, freq_bins, time_bins)
            frequencies: 频率轴
            window_times: 每个窗口的开始时间
        """
        print(f"\n处理 {window_sec}秒窗口...")
        
        # 提取窗口
        windows, window_times = self.extract_windows(window_sec)
        
        # 计算每个窗口的STFT
        stft_windows = []
        freq_bins = None
        
        for i, window in enumerate(windows):
            channel_stfts = []
            for ch in range(window.shape[0]):  # 对每个通道
                stft_db, freqs, times = self.compute_stft(window[ch, :])
                channel_stfts.append(stft_db)
            
            # 堆叠通道
            window_stft = np.array(channel_stfts)  # (22, freq_bins, time_bins)
            stft_windows.append(window_stft)
            
            if freq_bins is None:
                freq_bins = len(freqs)
                print(f"  频率分辨率: {freq_bins} bins")
                print(f"  时间分辨率: {len(times)} bins/窗口")
        
        stft_data = np.array(stft_windows)  # (n_windows, 22, freq_bins, time_bins)
        print(f"  STFT数据形状: {stft_data.shape}")
        
        return stft_data, freqs, window_times
    
    def process_all_windows(self):
        """
        处理所有窗口长度
        """
        results = {}
        
        for window_sec in self.window_lengths:
            print(f"\n{'─'*50}")
            
            # 转换为STFT
            stft_data, freqs, window_times = self.convert_to_stft(window_sec)
            
            # 保存结果
            output_file = self.output_dir / f"{Path(self.npz_file).stem}_stft_{window_sec}s.npz"
            np.savez_compressed(
                output_file,
                stft_data=stft_data,
                frequencies=freqs,
                window_times=window_times,
                channels=self.channels,
                sfreq=self.sfreq,
                window_sec=window_sec,
                nperseg=self.nperseg,
                noverlap=self.noverlap
            )
            
            results[window_sec] = {
                'file': str(output_file),
                'shape': stft_data.shape,
                'freq_range': [freqs[0], freqs[-1]],
                'n_windows': len(stft_data)
            }
            
            print(f"\n✅ 已保存: {output_file}")
            print(f"   数据形状: {stft_data.shape}")
            print(f"   频率范围: {freqs[0]:.1f} - {freqs[-1]:.1f} Hz")
        
        return results
    
    def visualize_stft(self, window_sec, channel_idx=0, window_idx=0):
        """
        可视化STFT结果
        """
        import matplotlib.pyplot as plt
        
        # 重新计算STFT（或从保存的文件加载）
        stft_data, freqs, window_times = self.convert_to_stft(window_sec)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'STFT时频图 (窗口={window_sec}秒, 通道={self.channels[channel_idx]})', 
                     fontsize=14)
        
        # 获取当前窗口的STFT
        current_stft = stft_data[window_idx, channel_idx, :, :]
        
        # 子图1: STFT频谱图
        im1 = axes[0, 0].imshow(
            current_stft, 
            aspect='auto',
            origin='lower',
            extent=[window_times[window_idx], 
                   window_times[window_idx] + window_sec,
                   freqs[0], freqs[-1]],
            cmap='jet'
        )
        axes[0, 0].set_xlabel('时间 (秒)')
        axes[0, 0].set_ylabel('频率 (Hz)')
        axes[0, 0].set_title('STFT频谱图')
        plt.colorbar(im1, ax=axes[0, 0], label='功率 (dB)')
        
        # 子图2: 平均频谱
        avg_spectrum = np.mean(current_stft, axis=1)
        axes[0, 1].plot(freqs, avg_spectrum)
        axes[0, 1].set_xlabel('频率 (Hz)')
        axes[0, 1].set_ylabel('平均功率 (dB)')
        axes[0, 1].set_title('平均频谱')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlim([0, 70])
        
        # 子图3: 时间演变（特定频带）
        freq_bands = {
            'Delta (0.5-4 Hz)': (0.5, 4),
            'Theta (4-8 Hz)': (4, 8),
            'Alpha (8-13 Hz)': (8, 13),
            'Beta (13-30 Hz)': (13, 30),
            'Gamma (30-70 Hz)': (30, 70)
        }
        
        for band_name, (low, high) in freq_bands.items():
            mask = (freqs >= low) & (freqs <= high)
            band_power = np.mean(current_stft[mask, :], axis=0)
            time_axis = np.linspace(0, window_sec, len(band_power))
            axes[1, 0].plot(time_axis, band_power, label=band_name)
        
        axes[1, 0].set_xlabel('时间 (秒)')
        axes[1, 0].set_ylabel('平均功率 (dB)')
        axes[1, 0].set_title('各频带时间演变')
        axes[1, 0].legend(loc='upper right', fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 子图4: 原始信号
        window_samples = int(window_sec * self.sfreq)
        start_sample = window_idx * (window_samples // 2)
        end_sample = start_sample + window_samples
        original_signal = self.eeg_data[channel_idx, start_sample:end_sample]
        time_axis = np.linspace(0, window_sec, len(original_signal))
        
        axes[1, 1].plot(time_axis, original_signal)
        axes[1, 1].set_xlabel('时间 (秒)')
        axes[1, 1].set_ylabel('幅值 (μV)')
        axes[1, 1].set_title('原始信号')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'stft_visualization_{window_sec}s.png', dpi=150)
        plt.show()
        
        return fig


# ========== 2. PyTorch数据集类（用于神经网络） ==========

class STFTDataset(Dataset):
    """
    STFT时频图数据集，可直接用于神经网络
    """
    
    def __init__(self, stft_file, model_type='cnn', transform=None):
        """
        参数:
            stft_file: 保存的STFT .npz文件
            model_type: 'cnn', 'lstm', 'transformer'
            transform: 数据增强变换
        """
        print(f"\n加载STFT数据集: {stft_file}")
        
        # 加载数据
        data = np.load(stft_file, allow_pickle=True)
        self.stft_data = data['stft_data']  # (n_windows, n_channels, freq_bins, time_bins)
        self.frequencies = data['frequencies']
        self.window_times = data['window_times']
        self.channels = data['channels']
        self.window_sec = data['window_sec']
        self.model_type = model_type
        self.transform = transform
        
        print(f"STFT数据形状: {self.stft_data.shape}")
        print(f"频率范围: {self.frequencies[0]:.1f} - {self.frequencies[-1]:.1f} Hz")
        print(f"窗口数量: {len(self.stft_data)}")
        
        # 数据归一化
        self._normalize()
        
    def _normalize(self):
        """归一化STFT数据"""
        # 对每个通道独立归一化
        self.mean = np.mean(self.stft_data, axis=(0, 2, 3), keepdims=True)
        self.std = np.std(self.stft_data, axis=(0, 2, 3), keepdims=True) + 1e-8
        self.stft_data = (self.stft_data - self.mean) / self.std
        
    def __len__(self):
        return len(self.stft_data)
    
    def __getitem__(self, idx):
        # 获取当前窗口
        stft = self.stft_data[idx]  # (n_channels, freq_bins, time_bins)
        
        # 根据模型类型调整格式
        if self.model_type == 'cnn':
            # CNN输入: (channels, freq, time) 或添加通道维度
            # 对于2D CNN，可以处理为 (freq, time, channels)
            # 这里我们保持 (channels, freq, time)，让模型处理
            data = torch.FloatTensor(stft)
            
        elif self.model_type == 'lstm':
            # LSTM输入: (time_steps, features)
            # 将频率和通道合并为特征
            n_channels, n_freq, n_time = stft.shape
            data = stft.transpose(2, 0, 1).reshape(n_time, -1)  # (time, channels*freq)
            data = torch.FloatTensor(data)
            
        elif self.model_type == 'transformer':
            # Transformer输入: (seq_len, features)
            # 类似LSTM
            n_channels, n_freq, n_time = stft.shape
            data = stft.transpose(2, 0, 1).reshape(n_time, -1)
            data = torch.FloatTensor(data)
            
        elif self.model_type == 'spectrogram':
            # 作为单通道图像 (freq, time)
            # 取所有通道的平均
            data = np.mean(stft, axis=0)  # (freq, time)
            data = torch.FloatTensor(data[np.newaxis, :, :])  # (1, freq, time)
            
        else:
            data = torch.FloatTensor(stft)
        
        # 数据增强
        if self.transform:
            data = self.transform(data)
        
        return data

def parse_chb_summary(txt_file_path):
    """
    从summary.txt文件解析发作时间
    """
    print(f"\n从文件加载发作时间: {txt_file_path}")
    print("="*60)
    
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return {}, 256
    
    # 解析采样率
    rate_match = re.search(r'Data Sampling Rate:\s*(\d+)\s*Hz', content)
    sampling_rate = int(rate_match.group(1)) if rate_match else 256
    print(f"采样率: {sampling_rate} Hz")
    
    # 解析发作时间
    seizure_times = {}
    
    # 按文件分割
    file_pattern = r'File Name:\s*(.+?\.edf).*?Number of Seizures in File:\s*(\d+)(.*?)(?=File Name:|\Z)'
    file_matches = re.findall(file_pattern, content, re.DOTALL | re.IGNORECASE)
    
    print(f"找到 {len(file_matches)} 个文件记录")
    
    for file_name, n_seizures_str, file_content in file_matches:
        file_name = file_name.strip()
        n_seizures = int(n_seizures_str)
        
        if n_seizures > 0:
            # 解析发作时间
            seizure_pattern = r'Seizure Start Time:\s*(\d+)\s*seconds.*?Seizure End Time:\s*(\d+)\s*seconds'
            seizures = re.findall(seizure_pattern, file_content, re.DOTALL | re.IGNORECASE)
            
            times = []
            for start, end in seizures:
                start_time = int(start)
                end_time = int(end)
                times.append((start_time, end_time))
                print(f"  {file_name}: {start_time}-{end_time}秒")
            
            seizure_times[file_name] = times
    
    print(f"\n总共找到 {len(seizure_times)} 个包含发作的文件")
    return seizure_times, sampling_rate
# ========== 3. 批处理多个文件 ==========

def batch_process_stft(input_dir='extracted_data', output_dir='stft_data', 
                       window_lengths=[5, 15, 30], seizure_times=None):
    """
    批量处理多个.npz文件
    
    参数:
        input_dir: 输入目录
        output_dir: 输出目录
        window_lengths: 窗口长度列表
        seizure_times: 发作时间字典，键为文件名，值为发作时间列表
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 找到所有.npz文件
    npz_files = list(input_path.glob('**/*full.npz'))
    print(f"找到 {len(npz_files)} 个.npz文件")
    
    # 计算所有样本的发作前期和发作间期总时长
    global_preictal_duration = 0
    global_interictal_duration = 0
    
    print(f"\n{'='*60}")
    print("计算全局发作前期和发作间期总时长")
    print(f"{'='*60}")
    
    # 首先收集所有发作时间
    all_seizure_times = {}
    for npz_file in npz_files:
        patient_dir = npz_file.parent.name
        file_stem = npz_file.stem
        edf_file_name = file_stem.replace('_full', '.edf')
        
        summary_file = f"D:/Graduation_thesis/data/chb-mit-scalp-eeg-database-1.0.0/{patient_dir}/{patient_dir}-summary.txt"
        file_seizure_times, _ = parse_chb_summary(summary_file)
        all_seizure_times.update(file_seizure_times)
    
    # 为每个患者收集所有文件的时间信息
    patient_files = {}
    for npz_file in npz_files:
        patient_dir = npz_file.parent.name
        file_stem = npz_file.stem
        edf_file_name = file_stem.replace('_full', '.edf')
        
        # 从summary.txt获取文件信息
        summary_file = f"D:/Graduation_thesis/data/chb-mit-scalp-eeg-database-1.0.0/{patient_dir}/{patient_dir}-summary.txt"
        
        # 读取整个文件内容
        with open(summary_file, 'r') as f:
            content = f.read()
        
        # 使用正则表达式提取文件信息
        file_pattern = rf'File Name:\s*{edf_file_name}.*?File Start Time:\s*(\d+):(\d+):(\d+).*?File End Time:\s*(\d+):(\d+):(\d+).*?Number of Seizures in File:\s*(\d+)(.*?)(?=File Name:|\Z)'
        file_match = re.search(file_pattern, content, re.DOTALL | re.IGNORECASE)
        
        if file_match:
            # 提取开始时间
            start_h, start_m, start_s = map(int, file_match.groups()[0:3])
            file_start_time = start_h * 3600 + start_m * 60 + start_s
            
            # 提取结束时间
            end_h, end_m, end_s = map(int, file_match.groups()[3:6])
            file_end_time = end_h * 3600 + end_m * 60 + end_s
            
            # 处理跨天情况（结束时间小于开始时间）
            if file_end_time < file_start_time:
                file_end_time += 24 * 3600  # 加一天
            
            # 计算时长
            total_duration = file_end_time - file_start_time
            
            # 提取发作时间
            n_seizures = int(file_match.group(7))
            file_content = file_match.group(8)
            
            if n_seizures > 0:
                seizure_pattern = r'Seizure Start Time:\s*(\d+)\s*seconds.*?Seizure End Time:\s*(\d+)\s*seconds'
                seizures = re.findall(seizure_pattern, file_content, re.DOTALL | re.IGNORECASE)
                # 将相对时间转换为绝对时间
                file_seizure_times = [(file_start_time + int(start), file_start_time + int(end)) for start, end in seizures]
            else:
                file_seizure_times = []
        else:
            # 如果正则匹配失败，使用后备方案
            # 加载数据以获取总时长
            print("使用备用方案")
            data = np.load(npz_file)
            times = data['times']
            total_duration = times[-1]
            file_start_time = 0
            file_end_time = total_duration
            file_seizure_times = all_seizure_times.get(edf_file_name, [])
        
        if patient_dir not in patient_files:
            patient_files[patient_dir] = []
        
        patient_files[patient_dir].append({
            'file_name': edf_file_name,
            'npz_file': npz_file,
            'start_time': file_start_time,
            'end_time': file_end_time,
            'duration': total_duration,
            'seizure_times': file_seizure_times
        })
    
    # 对每个患者的文件按时间排序并处理跨天连续性
    for patient_dir in patient_files:
        files = patient_files[patient_dir]
        # 按文件名排序（确保顺序正确）
        files.sort(key=lambda x: x['file_name'])
        
        # 处理跨天连续性
        previous_end_time = 0
        for i, file_info in enumerate(files):
            if i == 0:
                # 第一个文件保持不变
                previous_end_time = file_info['end_time']
            else:
                current_start = file_info['start_time']
                # 计算时间差
                time_diff = previous_end_time - current_start
                
                # 如果当前文件的开始时间小于前一个文件的结束时间，说明跨天了
                if time_diff > 0:
                    # 调整当前文件的时间，使其与前一个文件连续
                    file_info['start_time'] = previous_end_time
                    file_info['end_time'] = previous_end_time + (file_info['end_time'] - current_start)
                    
                    # 调整发作时间
                    adjusted_seizures = []
                    for start, end in file_info['seizure_times']:
                        adjusted_start = previous_end_time + (start - current_start)
                        adjusted_end = previous_end_time + (end - current_start)
                        adjusted_seizures.append((adjusted_start, adjusted_end))
                    file_info['seizure_times'] = adjusted_seizures
                
                previous_end_time = file_info['end_time']
        
        # 再次按时间排序
        files.sort(key=lambda x: x['start_time'])
        
        # 打印文件时间信息
        print(f"\n{patient_dir} 文件时间信息:")
        for file_info in files:
            print(f"  {file_info['file_name']}: {file_info['start_time']/3600:.2f} - {file_info['end_time']/3600:.2f}小时")
    
    # 计算所有文件的有效时间段
    for patient_dir, files in patient_files.items():
        # 获取该患者的所有发作时间
        patient_seizures = []
        for file_info in files:
            patient_seizures.extend(file_info['seizure_times'])
        # 按时间排序
        patient_seizures.sort(key=lambda x: x[0])
        
        # 计算该患者的所有有效时间段
        patient_valid_ranges = []
        
        if patient_seizures:
            # 处理第一个发作
            first_seizure_start = patient_seizures[0][0]
            
            # 第一个发作前的发作间期（发作前35分钟之前的时间段）
            interictal_end = first_seizure_start - 35 * 60
            if interictal_end > 0:
                # 患者最早的文件开始时间
                earliest_file_start = min([f['start_time'] for f in files])
                interictal_start = max(earliest_file_start, 0)
                if interictal_start < interictal_end:
                    patient_valid_ranges.append(('interictal', interictal_start, interictal_end))
            
            # 第一个发作前的发作前期（如果有）
            pre_seizure_start = max(0, first_seizure_start - 35 * 60)
            sph_start = first_seizure_start - 5 * 60
            
            if pre_seizure_start < sph_start:
                patient_valid_ranges.append(('preictal', pre_seizure_start, sph_start))
            
            # 处理发作间期
            for i in range(len(patient_seizures) - 1):
                current_seizure_end = patient_seizures[i][1]
                next_seizure_start = patient_seizures[i+1][0]
                
                # 发作间期：当前发作结束后30分钟至下一次发作开始前30分钟
                interictal_start = current_seizure_end + 30 * 60
                interictal_end = next_seizure_start - 30 * 60
                
                if interictal_start < interictal_end:
                    patient_valid_ranges.append(('interictal', interictal_start, interictal_end))
                
                # 下一次发作的发作前期（排除SPH）
                pre_seizure_start = max(interictal_end, next_seizure_start - 35 * 60)
                sph_start = next_seizure_start - 5 * 60
                
                if pre_seizure_start < sph_start:
                    patient_valid_ranges.append(('preictal', pre_seizure_start, sph_start))
            
            # 处理最后一个发作后的发作间期
            last_seizure_end = patient_seizures[-1][1]
            # 计算患者最后一个文件的结束时间
            last_file_end = max([f['end_time'] for f in files])
            interictal_start = last_seizure_end + 30 * 60
            if interictal_start < last_file_end:
                patient_valid_ranges.append(('interictal', interictal_start, last_file_end))
        
        # 为每个文件计算有效时间段
        for file_info in files:
            file_start = file_info['start_time']
            file_end = file_info['end_time']
            
            # 计算该文件内的有效时间段
            file_valid_ranges = []
            for range_type, range_start, range_end in patient_valid_ranges:
                # 计算文件与有效范围的交集
                overlap_start = max(file_start, range_start)
                overlap_end = min(file_end, range_end)
                
                if overlap_start < overlap_end:
                    # 计算在文件内的相对时间
                    relative_start = overlap_start - file_start
                    relative_end = overlap_end - file_start
                    file_valid_ranges.append((range_type, relative_start, relative_end))
            
            # 计算该文件的发作前期和发作间期时长
            file_preictal = 0
            file_interictal = 0
            
            for range_type, start, end in file_valid_ranges:
                duration = end - start
                if range_type == 'preictal':
                    file_preictal += duration
                elif range_type == 'interictal':
                    file_interictal += duration
            
            # 更新全局时长
            global_preictal_duration += file_preictal
            global_interictal_duration += file_interictal
            
            # 存储文件的有效时间段信息
            file_info['valid_ranges'] = file_valid_ranges
    
    print(f"全局发作前期总时长: {global_preictal_duration:.1f}秒")
    print(f"全局发作间期总时长: {global_interictal_duration:.1f}秒")
    
    all_results = {}
    
    for npz_file in npz_files:
        patient_dir = npz_file.parent.name
        file_stem = npz_file.stem
        edf_file_name = file_stem.replace('_full', '.edf')
        
        summary_file = f"D:/Graduation_thesis/data/chb-mit-scalp-eeg-database-1.0.0/{patient_dir}/{patient_dir}-summary.txt"
        file_seizure_times, _ = parse_chb_summary(summary_file)
        # 获取当前文件的发作时间
        current_file_seizure_times = file_seizure_times.get(edf_file_name, None)
        
        print(f"\n{'#'*60}")
        print(f"处理: {patient_dir}/{file_stem}")
        print(f"{'#'*60}")
        
        # 找到当前文件的信息
        current_file_info = None
        for file_info in patient_files[patient_dir]:
            if file_info['npz_file'] == npz_file:
                current_file_info = file_info
                break
        
        # 提取有效时间段信息
        valid_ranges = None
        if current_file_info and 'valid_ranges' in current_file_info:
            # 保留类型信息
            valid_ranges = current_file_info['valid_ranges']
        
        # 创建转换器
        converter = EEG2STFTConverter(
            npz_file=str(npz_file),
            sfreq=256,
            window_lengths=window_lengths,
            nperseg=256,
            noverlap=128,
            output_dir=str(output_path / patient_dir),
            seizure_times=current_file_seizure_times,
            global_preictal_duration=global_preictal_duration,
            global_interictal_duration=global_interictal_duration,
            valid_ranges=valid_ranges
        )
        
        # 处理所有窗口
        results = converter.process_all_windows()
        all_results[f"{patient_dir}/{file_stem}"] = results
    
    # 保存处理记录
    import json
    with open(output_path / 'processing_log.json', 'w') as f:
        # 转换numpy类型为Python原生类型
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        serializable_results = {}
        for key, value in all_results.items():
            serializable_results[key] = {
                k: {kk: convert_to_serializable(vv) for kk, vv in v.items()}
                for k, v in value.items()
            }
        
        json.dump(serializable_results, f, indent=2)
    
    return all_results


if __name__ == "__main__":
    # 定义22个标准通道
    standard_22_channels = [
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
        'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
        'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
        'FZ-CZ', 'CZ-PZ',
        'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8'
    ]
    
    # 定义有效患者（排除chb12和13）
    valid_patients = [f"chb{i:02d}" for i in range(1, 4) if i not in [12, 13]]
    #valid_patients = [f"chb{i:02d}" for i in range(3, 3) ]
    # 数据路径
    data_path = "D:/Graduation_thesis/data/chb-mit-scalp-eeg-database-1.0.0"
    
   
    
    #取消下面的注释来提取所有患者
    # extract_all_patients_22channels(
    #     data_path, 
    #     standard_22_channels, 
    #     valid_patients
    # )
    # # 取消注释来重构患者数据
    # reconstruct_patient_data("chb01")
    # reconstruct_patient_data("chb02")
    # reconstruct_patient_data("chb03")
    
    


# 设置路径
    data_dir = "D:\Graduation_thesis\code\extracted_data"
    output_dir = "D:/Graduation_thesis/code/stft_data"
    
    # # 1. 处理单个文件示例
    # print("="*70)
    # print("示例1: 处理单个文件")
    # print("="*70)
    
    # npz_file = f"{data_dir}/chb01/chb01_01_full.npz"
    
    # if Path(npz_file).exists():
    #     converter = EEG2STFTConverter(
    #         npz_file=npz_file,
    #         sfreq=256,
    #         window_lengths=[5, 15, 30],  # 5秒、15秒、30秒窗口
    #         nperseg=256,  # 1秒窗口 @ 256Hz
    #         noverlap=128,  # 50%重叠
    #         output_dir=f"{output_dir}/chb01"
    #     )
        
    #     # 处理所有窗口
    #     results = converter.process_all_windows()
        
    #     # 可视化第一个窗口
    #     converter.visualize_stft(window_sec=5, channel_idx=0, window_idx=0)
    
    
    #3. 批量处理
    print("\n" + "="*70)
    print("示例: 批量处理所有文件")
    print("="*70)
    all_results = batch_process_stft(
        input_dir=data_dir,
        output_dir=output_dir,
        #window_lengths=[5, 15, 30]
        window_lengths=[30]

    )
    
    print("\n✅ STFT转换完成!")
    