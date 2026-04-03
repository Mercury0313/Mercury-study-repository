import json
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')


class LocalSTFTDataset(Dataset):
    """
    从本地加载STFT训练样本的数据集
    """
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data = torch.FloatTensor(self.samples[idx])
        label = torch.FloatTensor(self.labels[idx])
        return data, label


def load_training_samples(save_path):
    """
    加载训练样本
    """
    print(f"\n加载训练样本: {save_path}")
    
    with np.load(save_path, allow_pickle=True) as data:
        samples = data['data']
        labels = data['labels']
        metadata = data['metadata'] if 'metadata' in data else []
        global_mean = data['global_mean'] if 'global_mean' in data else None
        global_std = data['global_std'] if 'global_std' in data else None
    
    print(f"  加载成功: {len(samples)} 个样本")
    print(f"  数据形状: {samples.shape}")
    print(f"  标签形状: {labels.shape}")
    
    return samples, labels, metadata, global_mean, global_std


class MultiFileSTFTDataset(Dataset):
    """
    多文件STFT数据集（带错误处理）
    """
    def __init__(self, stft_files, window_size=None, stride=None, 
                 transform=None, normalize=True):
        self.stft_files = stft_files
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.normalize = normalize
        
        print(f"\n{'='*60}")
        print(f"加载多文件STFT数据集")
        print(f"{'='*60}")
        print(f"文件数量: {len(stft_files)}")
        
        self.samples = []
        self.file_metadata = []
        self.file_original_times = {}
        self.corrupted_files = []
        
        for file_idx, stft_file in enumerate(stft_files):
            try:
                file_samples = self._load_file(stft_file, file_idx)
                if file_samples:  # 只有当成功加载样本时才添加
                    self.samples.extend(file_samples)
            except Exception as e:
                print(f"\n❌ 警告: 无法加载文件 {stft_file.name}: {str(e)}")
                self.corrupted_files.append(str(stft_file))
                continue
            
        print(f"\n成功加载样本数: {len(self.samples)}")
        if self.corrupted_files:
            print(f"损坏文件数: {len(self.corrupted_files)}")
        
        if len(self.samples) == 0:
            raise ValueError("没有成功加载任何样本！")
        
        # 初始化标准化参数为None
        self.global_mean = None
        self.global_std = None
        
        if normalize:
            self._compute_global_stats()
    
    def _verify_file(self, stft_file):
        """验证文件是否完整"""
        try:
            # 尝试打开文件并检查关键键值
            with np.load(stft_file, allow_pickle=True) as data:
                required_keys = ['stft_data', 'window_times', 'sfreq', 'window_sec']
                for key in required_keys:
                    if key not in data:
                        print(f"  文件缺少必要键: {key}")
                        return False
                return True
        except Exception as e:
            print(f"  文件验证失败: {e}")
            return False
    
    def _load_file(self, stft_file, file_idx):
        """加载单个STFT文件（带错误处理）- 保持4D形状"""
        print(f"\n加载: {stft_file.name}")
        
        # 先验证文件
        if not self._verify_file(stft_file):
            print(f"  ⚠️ 文件 {stft_file.name} 验证失败，跳过")
            return []
        
        try:
            # 使用上下文管理器确保文件正确关闭
            with np.load(stft_file, allow_pickle=True) as data:
                stft_data = data['stft_data']
                window_times = data['window_times']
                sfreq = data['sfreq']
                window_sec = data['window_sec']
                
                n_windows, n_channels, n_freq, n_time = stft_data.shape
                
                # 验证数据形状
                if n_windows == 0 or n_time == 0:
                    print(f"  ⚠️ 文件 {stft_file.name} 数据为空，跳过")
                    return []
                
                time_per_bin = window_sec / n_time
                
                print(f"  原始形状: {stft_data.shape}")
                print(f"  窗口起始时间范围: {window_times[0]:.1f} - {window_times[-1]:.1f}秒")
                print(f"  每个时间bin: {time_per_bin:.3f}秒")
                
                # 转置为 (n_windows, n_channels, n_time, n_freq) 以匹配要求的输入格式
                stft_data_transposed = stft_data.transpose(0, 1, 3, 2)  # (n_windows, n_channels, n_time, n_freq)
                
                print(f"  转置后形状: {stft_data_transposed.shape}")
                
                # 检查数据中是否有NaN或Inf
                if np.isnan(stft_data_transposed).any() or np.isinf(stft_data_transposed).any():
                    print(f"  ⚠️ 文件 {stft_file.name} 包含NaN或Inf，尝试修复")
                    stft_data_transposed = np.nan_to_num(stft_data_transposed, nan=0.0, posinf=0.0, neginf=0.0)
                
                # 存储时间信息
                self.file_original_times[file_idx] = {
                    'window_times': window_times,
                    'time_per_bin': time_per_bin,
                    'window_sec': window_sec,
                    'n_time': n_time,
                    'n_freq': n_freq,
                    'file_name': stft_file.name
                }
                
                # 创建样本
                samples = []
                
                if self.window_size is None:
                    # 使用整个STFT窗口
                    for w in range(n_windows):
                        mid_time = window_times[w] + window_sec/2
                        samples.append({
                            'file_idx': file_idx,
                            'window_idx': w,
                            't_start': 0,
                            't_end': n_time,
                            'data': stft_data_transposed[w],  # 形状: (n_channels, n_time, n_freq)
                            'time': mid_time,
                            'time_start': window_times[w],
                            'time_end': window_times[w] + window_sec
                        })
                else:
                    # 滑动窗口（在时间维度上滑动）
                    for w in range(n_windows):
                        for t_start in range(0, n_time - self.window_size + 1, self.stride):
                            t_end = t_start + self.window_size
                            
                            sub_start_time = window_times[w] + t_start * time_per_bin
                            sub_end_time = window_times[w] + t_end * time_per_bin
                            sub_mid_time = (sub_start_time + sub_end_time) / 2
                            
                            # 提取时间维度的切片，保持 (n_channels, window_size, n_freq) 形状
                            window_data = stft_data_transposed[w, :, t_start:t_end, :]
                            
                            # 检查窗口数据
                            if window_data.size == 0:
                                continue
                                
                            samples.append({
                                'file_idx': file_idx,
                                'window_idx': w,
                                't_start': t_start,
                                't_end': t_end,
                                'data': window_data,  # 形状: (n_channels, window_size, n_freq)
                                'time': sub_mid_time,
                                'time_start': sub_start_time,
                                'time_end': sub_end_time
                            })
                
                # 保存元数据
                self.file_metadata.append({
                    'file': str(stft_file),
                    'n_windows': n_windows,
                    'n_time': n_time,
                    'n_freq': n_freq,
                    'sfreq': sfreq,
                    'window_sec': window_sec
                })
                
                return samples
                
        except Exception as e:
            print(f"  ❌ 加载文件 {stft_file.name} 时出错: {str(e)}")
            return []
    
    def _compute_global_stats(self):
        """计算全局标准化参数 - 适应4D数据"""
        print("\n计算全局标准化参数...")
        
        # 如果样本太多，只使用部分样本计算
        n_samples = min(1000, len(self.samples))
        if n_samples == 0:
            print("  ⚠️ 没有样本用于计算统计量，使用默认值")
            # 使用正确的形状创建默认值
            sample_data = self.samples[0]['data']
            self.global_mean = np.zeros((1, sample_data.shape[0], 1, 1), dtype=np.float32)
            self.global_std = np.ones((1, sample_data.shape[0], 1, 1), dtype=np.float32)
            return
            
        sample_indices = np.random.choice(len(self.samples), n_samples, replace=False)
        
        # 收集数据
        all_data = []
        for idx in sample_indices:
            all_data.append(self.samples[idx]['data'])
        
        try:
            # 尝试拼接所有数据
            if len(all_data) > 0:
                # 检查数据形状是否一致
                shapes = [d.shape for d in all_data]
                if len(set(shapes)) > 1:
                    print(f"  警告: 数据形状不一致: {set(shapes)}")
                    # 使用第一个样本的形状作为标准
                    target_shape = all_data[0].shape
                    # 调整所有数据到相同形状（如果需要的话）
                    adjusted_data = []
                    for d in all_data:
                        if d.shape != target_shape:
                            # 简单的截断或填充
                            d_adjusted = np.zeros(target_shape, dtype=np.float32)
                            min_c = min(d.shape[0], target_shape[0])
                            min_t = min(d.shape[1], target_shape[1])
                            min_f = min(d.shape[2], target_shape[2])
                            d_adjusted[:min_c, :min_t, :min_f] = d[:min_c, :min_t, :min_f]
                            adjusted_data.append(d_adjusted)
                        else:
                            adjusted_data.append(d)
                    all_data = adjusted_data
                
                # 堆叠数据: (n_samples, n_channels, n_time, n_freq)
                all_data_array = np.stack(all_data, axis=0)
                
                # 计算均值和标准差，保持维度用于广播
                self.global_mean = np.mean(all_data_array, axis=(0, 2, 3), keepdims=True).astype(np.float32)
                self.global_std = np.std(all_data_array, axis=(0, 2, 3), keepdims=True).astype(np.float32) + 1e-8
                
                print(f"  均值形状: {self.global_mean.shape}")
                print(f"  标准差形状: {self.global_std.shape}")
            else:
                raise ValueError("没有有效数据")
                
        except Exception as e:
            print(f"  ⚠️ 计算统计量时出错: {e}")
            # 使用默认值
            print("  使用默认标准化参数")
            sample_data = self.samples[0]['data']
            self.global_mean = np.zeros((1, sample_data.shape[0], 1, 1), dtype=np.float32)
            self.global_std = np.ones((1, sample_data.shape[0], 1, 1), dtype=np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = sample['data'].copy()  # 形状: (n_channels, n_time, n_freq)
        
        # 确保通道数一致（目标通道数为22）
        target_channels = 22
        if data.shape[0] != target_channels:
            if data.shape[0] < target_channels:
                # 填充通道
                pad_channels = target_channels - data.shape[0]
                data = np.pad(data, ((0, pad_channels), (0, 0), (0, 0)), mode='constant')
            else:
                # 截断通道
                data = data[:target_channels, :, :]
        
        # 应用标准化
        if self.normalize and self.global_mean is not None and self.global_std is not None:
            # 确保数据维度匹配
            if data.shape[0] != self.global_mean.shape[1]:
                print(f"警告: 通道数不匹配: {data.shape[0]} vs {self.global_mean.shape[1]}")
                # 调整global_mean和global_std的通道数
                if data.shape[0] < self.global_mean.shape[1]:
                    self.global_mean = self.global_mean[:, :data.shape[0], :, :]
                    self.global_std = self.global_std[:, :data.shape[0], :, :]
                else:
                    pad_channels = data.shape[0] - self.global_mean.shape[1]
                    self.global_mean = np.pad(self.global_mean, ((0, 0), (0, pad_channels), (0, 0), (0, 0)), mode='constant')
                    self.global_std = np.pad(self.global_std, ((0, 0), (0, pad_channels), (0, 0), (0, 0)), mode='constant', constant_values=1)
            
            data = (data - self.global_mean[0]) / self.global_std[0]
        
        # 确保数据是float32类型
        data = data.astype(np.float32)
        
        # 返回形状: (n_channels, n_time, n_freq)
        return torch.FloatTensor(data)
    
    def get_metadata(self, idx):
        return self.samples[idx]


class MultiFileSTFTWithLabels(Dataset):
    """
    带标签的多文件STFT数据集（修复版）
    """
    def __init__(self, stft_files, seizure_times, window_size=None, stride=None,
                 normalize=True, balance_classes=True):
        
        # 初始化基础数据集
        print("\n初始化基础数据集...")
        try:
            self.base_dataset = MultiFileSTFTDataset(
                stft_files, window_size, stride, normalize=normalize
            )
        except Exception as e:
            print(f"❌ 初始化基础数据集失败: {e}")
            raise
        
        self.seizure_times = seizure_times
        
        # 创建文件名映射
        print("\n创建文件映射...")
        self.file_idx_to_name = {}
        self.valid_files = set()
        
        for idx, stft_file in enumerate(stft_files):
            # 从STFT文件名提取原始EDF文件名
            base_name = stft_file.name.replace('_full_stft_30s.npz', '.edf')
            # 也尝试其他可能的命名格式
            if base_name == stft_file.name:
                base_name = stft_file.name.replace('_stft_30s.npz', '.edf')
            
            self.file_idx_to_name[idx] = base_name
            
            # 检查是否匹配发作文件
            if base_name in seizure_times:
                times = seizure_times[base_name]
                print(f"  ✅ 索引 {idx:2d}: {base_name} -> 发作时间: {times}")
                self.valid_files.add(base_name)
            else:
                print(f"  ❌ 索引 {idx:2d}: {base_name} -> 无发作")
        
        # 计算所有发作时间（按时间排序）
        all_seizure_times = []
        for file_name, times in seizure_times.items():
            all_seizure_times.extend(times)
        all_seizure_times.sort(key=lambda x: x[0])
        
        # 生成标签
        print("\n生成标签...")
        self.labels = []
        self.label_info = []
        
        preictal_count = 0  # 发作前期
        interictal_count = 0  # 发作间期
        
        # 检查是否有样本
        if len(self.base_dataset) == 0:
            print("\n❌ 错误: 基础数据集中没有样本！")
            raise ValueError("Empty dataset")
        
        for idx in range(len(self.base_dataset)):
            meta = self.base_dataset.get_metadata(idx)
            file_idx = meta['file_idx']
            time_start = meta['time_start']
            time_end = meta['time_end']
            
            file_name = self.file_idx_to_name.get(file_idx, "")
            
            # 判断是发作前期还是发作间期
            label = 0  # 默认发作间期
            matched_seizure = None
            skip_sample = False  # 是否跳过该样本
            
            # 检查是否在发作期间（应该被排除）
            if file_name in seizure_times:
                for seizure_start, seizure_end in seizure_times[file_name]:
                    # 如果窗口与发作期间重叠，跳过该样本
                    if (time_start <= seizure_end and time_end >= seizure_start):
                        skip_sample = True
                        break
            
            if skip_sample:
                continue  # 跳过发作期间的样本
            
            # 检查是否在发作后30分钟内（应该被排除）
            if file_name in seizure_times:
                for seizure_start, seizure_end in seizure_times[file_name]:
                    # 发作后30分钟内
                    post_seizure_end = seizure_end + 30 * 60
                    # 如果窗口与发作后30分钟内重叠，跳过该样本
                    if (time_start <= post_seizure_end and time_end >= seizure_end):
                        skip_sample = True
                        break
            
            if skip_sample:
                continue  # 跳过发作后30分钟内的样本
            
            # 检查是否在SPH（发作前5分钟，应该被排除）
            if file_name in seizure_times:
                for seizure_start, seizure_end in seizure_times[file_name]:
                    # SPH：发作前5分钟内
                    sph_start = seizure_start - 5 * 60
                    # 如果窗口与SPH重叠，跳过该样本
                    if (time_start <= seizure_start and time_end >= sph_start):
                        skip_sample = True
                        break
            
            if skip_sample:
                continue  # 跳过SPH内的样本
            
            # 检查是否在发作前期（发作前35分钟到发作前5分钟）
            if file_name in seizure_times:
                for seizure_start, seizure_end in seizure_times[file_name]:
                    # 发作前期：发作前35分钟到发作前5分钟
                    preictal_start = max(0, seizure_start - 35 * 60)
                    preictal_end = seizure_start - 5 * 60
                    
                    # 如果窗口与发作前期重叠，标记为发作前期
                    if (time_start <= preictal_end and time_end >= preictal_start):
                        label = 1  # 发作前期
                        matched_seizure = (seizure_start, seizure_end)
                        break
            
            # 检查是否在发作间期（发作后30分钟至下一次发作开始前35分钟）
            if label == 0 and file_name in seizure_times:
                for seizure_start, seizure_end in seizure_times[file_name]:
                    # 发作间期：发作后30分钟至下一次发作开始前35分钟
                    interictal_end = seizure_start - 35 * 60
                    # 如果窗口在发作间期范围内，标记为发作间期
                    if (time_start >= 0 and time_end <= interictal_end):
                        label = 0  # 发作间期
                        break
            
            self.labels.append(label)
            self.label_info.append({
                'file_idx': file_idx,
                'file_name': file_name,
                'time_range': (time_start, time_end),
                'label': label,
                'matched_seizure': matched_seizure
            })
            
            if label == 1:
                preictal_count += 1
            else:
                interictal_count += 1
        
        self.labels = np.array(self.labels)
        
        # 统计
        print(f"\n类别分布:")
        print(f"  类别0 (发作间期): {interictal_count} ({interictal_count/len(self.labels)*100:.2f}%)")
        print(f"  类别1 (发作前期): {preictal_count} ({preictal_count/len(self.labels)*100:.2f}%)")
        
        # 显示发作前期样本示例
        if preictal_count > 0:
            print(f"\n发作前期样本示例:")
            preictal_indices = np.where(self.labels == 1)[0]
            for i in preictal_indices[:min(10, len(preictal_indices))]:
                info = self.label_info[i]
                print(f"  样本 {i}: 文件 {info['file_name']}, "
                      f"时间范围 {info['time_range'][0]:.1f}-{info['time_range'][1]:.1f}秒, "
                      f"匹配发作 {info['matched_seizure']}")
        else:
            print("\n⚠️ 警告: 没有找到任何发作前期样本！")
            print("请检查以下文件是否应该有发作:")
            for file_name in seizure_times.keys():
                print(f"  - {file_name}")
        
        # 平衡类别
        self.balance_classes = balance_classes
        if balance_classes and preictal_count > 0 and interictal_count > 0:
            self._create_balanced_indices()
        else:
            print("\n⚠️ 警告: 无法平衡类别，将使用所有样本")
            self.balanced_indices = np.arange(len(self.labels))
    
    def _create_balanced_indices(self):
        """创建平衡类别的索引"""
        indices_class0 = np.where(self.labels == 0)[0]
        indices_class1 = np.where(self.labels == 1)[0]
        
        min_len = min(len(indices_class0), len(indices_class1))
        
        print(f"\n平衡类别: 每类使用 {min_len} 个样本")
        
        # 随机选择样本
        np.random.seed(42)  # 固定随机种子
        class0_selected = np.random.choice(indices_class0, min_len, replace=False)
        class1_selected = np.random.choice(indices_class1, min_len, replace=False)
        
        self.balanced_indices = np.concatenate([class0_selected, class1_selected])
        np.random.shuffle(self.balanced_indices)
        
        print(f"平衡后总样本数: {len(self.balanced_indices)}")
    
    def __len__(self):
        if hasattr(self, 'balanced_indices'):
            return len(self.balanced_indices)
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        if hasattr(self, 'balanced_indices'):
            idx = self.balanced_indices[idx]
        
        data = self.base_dataset[idx]
        label = self.labels[idx]
        
        label_onehot = torch.zeros(2)
        label_onehot[label] = 1
        
        return data, label_onehot


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


def load_all_patients_data(patient_ids, base_data_dir, stft_subdir="stft_data"):
    """
    加载多个病人的STFT数据和发作时间（带错误处理）
    """
    all_stft_files = []
    all_seizure_times = {}
    failed_patients = []
    
    print("\n" + "="*70)
    print(f"加载病人数据: {patient_ids}")
    print("="*70)
    
    for patient_id in patient_ids:
        print(f"\n--- 处理病人 {patient_id} ---")
        
        # STFT文件路径
        stft_dir = Path(base_data_dir) / "code" / f"{stft_subdir}/{patient_id}"
        if not stft_dir.exists():
            print(f"⚠️ 警告: STFT目录不存在: {stft_dir}")
            failed_patients.append(patient_id)
            continue
            
        stft_files = sorted(stft_dir.glob("*_stft_30s.npz"))
        print(f"找到 {len(stft_files)} 个STFT文件")
        
        # 发作时间文件路径
        summary_file = Path(base_data_dir) / f"data/chb-mit-scalp-eeg-database-1.0.0/{patient_id}/{patient_id}-summary.txt"
        if not summary_file.exists():
            print(f"⚠️ 警告: summary文件不存在: {summary_file}")
            failed_patients.append(patient_id)
            continue
        
        try:    
            seizure_times, _ = parse_chb_summary(summary_file)
            
            # 添加到总列表
            all_stft_files.extend(stft_files)
            all_seizure_times.update(seizure_times)
            
            print(f"病人 {patient_id} 发作文件数: {len(seizure_times)}")
        except Exception as e:
            print(f"⚠️ 处理病人 {patient_id} 时出错: {e}")
            failed_patients.append(patient_id)
            continue
    
    print(f"\n总计:")
    print(f"  STFT文件总数: {len(all_stft_files)}")
    print(f"  发作文件总数: {len(all_seizure_times)}")
    if failed_patients:
        print(f"  处理失败的病人: {failed_patients}")
    
    return all_stft_files, all_seizure_times

class ResidualBlock(nn.Module):
    """
    残差块
    """
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        #  shortcut连接
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        x += residual
        x = self.relu(x)
        
        return x


class ChannelAttention(nn.Module):
    """
    通道注意力模块
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # 平均池化路径
        avg_out = self.avg_pool(x).view(batch_size, channels)
        avg_out = self.fc1(avg_out)
        avg_out = self.relu(avg_out)
        avg_out = self.fc2(avg_out)
        
        # 最大池化路径
        max_out = self.max_pool(x).view(batch_size, channels)
        max_out = self.fc1(max_out)
        max_out = self.relu(max_out)
        max_out = self.fc2(max_out)
        
        # 融合
        out = avg_out + max_out
        out = self.sigmoid(out).view(batch_size, channels, 1, 1)
        
        return x * out


class SpatialAttention(nn.Module):
    """
    空间注意力模块
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 通道维度的平均池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 通道维度的最大池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接
        out = torch.cat([avg_out, max_out], dim=1)
        # 卷积
        out = self.conv(out)
        # 激活
        out = self.sigmoid(out)
        
        return x * out


class RDANet(nn.Module):
    """
    RDANet模型 - 基于高光谱图像分类架构
    输入格式: (batch, channels, height, width, spectral)
    原始架构: 输入 (1, 22, 9, 114)
    
    对于脑电信号频谱图，需要调整输入格式：
    如果输入是 (batch, 22, n_time, n_freq)，则视为 (batch, 1, 22, n_time, n_freq)
    即把22个通道视为空间高度，n_time视为宽度，n_freq视为光谱维度
    """
    def __init__(self, in_channels=1, num_classes=2, dropout=0.3, 
                 spectral_bands=114, spatial_height=22, spatial_width=9):
        """
        Args:
            in_channels: 输入通道数（对于高光谱，通常为1）
            num_classes: 分类类别数
            dropout: dropout率
            spectral_bands: 光谱波段数
            spatial_height: 空间高度
            spatial_width: 空间宽度
        """
        super().__init__()
        
        self.spectral_bands = spectral_bands
        self.spatial_height = spatial_height
        self.spatial_width = spatial_width
        
        # ========== 第一阶段：3D卷积 + 3D池化 ==========
        # Conv3D: 输入 (batch, in_channels, H, W, D)
        # 输出通道数: 64
        self.conv3d = nn.Conv3d(in_channels, 64, kernel_size=(spatial_height, 3, 5), 
                                 stride=(1, 1, 2), padding=0, bias=False)
        self.bn3d = nn.BatchNorm3d(64)
        self.relu3d = nn.ReLU(inplace=True)
        
        # 3D池化: 只在光谱维度池化
        self.pool3d = nn.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2), ceil_mode=True)
        
        # ========== 第二阶段：ResNet 残差块（2D卷积） ==========
        # 注意：经过3D卷积和池化后，输出为 (batch, 64, 1, 7, 28)
        # Reshape后为 (batch, 64, 7, 28)
        
        # ResNet Block1: 64通道，空间尺寸 7x28，重复2次
        self.block1 = self._make_layer(64, 64, 2, stride=1, dropout=dropout)
        
        # ResNet Block2: 128通道，空间尺寸 4x14，重复2次
        self.block2 = self._make_layer(64, 128, 2, stride=2, dropout=dropout)
        
        # ResNet Block3: 256通道，空间尺寸 2x7，重复2次
        self.block3 = self._make_layer(128, 256, 2, stride=2, dropout=dropout)
        
        # ResNet Block4: 512通道，空间尺寸 1x4，重复2次
        self.block4 = self._make_layer(256, 512, 2, stride=2, dropout=dropout)
        
        # ========== 第三阶段：双自注意力 ==========
        self.dual_attention = DualSelfAttention(512)
        
        # ========== 第四阶段：分类头 ==========
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dropout):
        """创建残差块组"""
        layers = []
        
        # 第一个残差块（可能改变通道数和尺寸）
        layers.append(BasicBlock(in_channels, out_channels, stride, dropout))
        
        # 后续残差块（保持通道数和尺寸）
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1, dropout))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: 输入张量
               选项1: (batch, 1, H, W, D) 标准5D高光谱输入
               选项2: (batch, 22, n_time, n_freq) 脑电频谱图，需reshape为 (batch, 1, 22, n_time, n_freq)
        Returns:
            output: (batch, num_classes)
        """
        # 处理输入维度
        if x.dim() == 4:
            # 输入是 (batch, channels, height, width) 格式
            # 假设 channels=22 是空间高度，height是时间，width是频率
            batch, channels, height, width = x.shape
            # 重塑为 (batch, 1, spatial_height, spatial_width, spectral_bands)
            # 这里假设 channels=22 是空间高度，height是宽度，width是光谱维度
            x = x.unsqueeze(1)  # (batch, 1, 22, height, width)
        
        # 输入形状: (batch, 1, H, W, D)
        # H = 22 (空间高度)
        # W = n_time (时间/宽度)
        # D = n_freq (频率/光谱)
        
        # ========== 3D卷积 ==========
        x = self.conv3d(x)  # (batch, 64, 1, W_out, D_out)
        x = self.bn3d(x)
        x = self.relu3d(x)
        
        # ========== 3D池化 ==========
        x = self.pool3d(x)  # (batch, 64, 1, W_pool, D_pool)
        
        # ========== Reshape: 移除高度维度 ==========
        # 输入: (batch, 64, 1, H, W) -> (batch, 64, H, W)
        x = x.squeeze(2)  # (batch, 64, H, W)
        
        # ========== ResNet 残差块 ==========
        x = self.block1(x)  # (batch, 64, H1, W1)
        x = self.block2(x)  # (batch, 128, H2, W2)
        x = self.block3(x)  # (batch, 256, H3, W3)
        x = self.block4(x)  # (batch, 512, H4, W4)
        
        # ========== 双自注意力 ==========
        x = self.dual_attention(x)  # (batch, 512, H4, W4)
        
        # ========== 分类 ==========
        x = self.global_avg_pool(x)  # (batch, 512, 1, 1)
        x = self.classifier(x)  # (batch, num_classes)
        
        return x


class DualSelfAttention(nn.Module):
    """
    双自注意力模块：位置自注意力 + 通道自注意力
    输入: (batch, C, H, W)
    输出: (batch, C, H, W)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        
        # 位置注意力
        self.position_query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.position_key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.position_value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # 通道注意力
        self.channel_query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.channel_key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.channel_value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # 输出投影
        self.position_gamma = nn.Parameter(torch.zeros(1))
        self.channel_gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch, C, H, W = x.shape
        N = H * W  # 空间位置数
        
        # ========== 位置自注意力 ==========
        # 生成 Q, K, V
        pos_q = self.position_query(x).view(batch, -1, N).permute(0, 2, 1)  # (B, N, C/8)
        pos_k = self.position_key(x).view(batch, -1, N)  # (B, C/8, N)
        pos_v = self.position_value(x).view(batch, -1, N)  # (B, C, N)
        
        # 计算注意力图
        pos_attn = torch.bmm(pos_q, pos_k)  # (B, N, N)
        pos_attn = F.softmax(pos_attn, dim=-1)
        
        # 加权求和
        pos_out = torch.bmm(pos_v, pos_attn.permute(0, 2, 1))  # (B, C, N)
        pos_out = pos_out.view(batch, C, H, W)
        pos_out = self.position_gamma * pos_out + x  # 残差连接
        
        # ========== 通道自注意力 ==========
        # 生成 Q, K, V
        chan_q = self.channel_query(pos_out).view(batch, C, -1)  # (B, C, N)
        chan_k = self.channel_key(pos_out).view(batch, C, -1)  # (B, C, N)
        chan_v = self.channel_value(pos_out).view(batch, C, -1)  # (B, C, N)
        
        # 计算注意力图
        chan_attn = torch.bmm(chan_q, chan_k.permute(0, 2, 1))  # (B, C, C)
        chan_attn = F.softmax(chan_attn, dim=-1)
        
        # 加权求和
        chan_out = torch.bmm(chan_attn, chan_v)  # (B, C, N)
        chan_out = chan_out.view(batch, C, H, W)
        chan_out = self.channel_gamma * chan_out + pos_out  # 残差连接
        
        return chan_out


class BasicBlock(nn.Module):
    """
    基本残差块（2个3x3卷积）
    当通道数变化或步长变化时，shortcut使用1x1卷积
    """
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        
        # 第一个卷积层（可能下采样）
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        # shortcut连接
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        x += residual
        x = self.relu(x)
        
        return x


def test_model(model, test_loader, device='cuda'):
    """
    测试模型
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            _, true = torch.max(labels, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(true.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    # 计算敏感性和特异性
    # 敏感性 = TP / (TP + FN) = 召回率，表示发作样本被正确识别的比例
    # 特异性 = TN / (TN + FP)，表示非发作样本被正确识别的比例
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # 敏感性
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # 特异性
    
    # 计算AUC
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(all_labels, all_probs)
    
    print("\n" + "="*60)
    print("测试结果")
    print("="*60)
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}  (发作样本正确识别率)")
    print(f"Specificity: {specificity:.4f}  (非发作样本正确识别率)")
    print(f"AUC:         {auc:.4f}")
    print("\n混淆矩阵:")
    print(f"         预测非发作  预测发作")
    print(f"实际非发作:   {cm[0,0]:5d}      {cm[0,1]:5d}")
    print(f"实际发作:     {cm[1,0]:5d}      {cm[1,1]:5d}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'auc': auc,
        'confusion_matrix': cm
    }


def main():
    print("\n" + "="*70)
    print("使用预处理好的样本验证模型效果")
    print("="*70)
    
    try:
        # 1. 设置基础路径
        base_data_dir = "D:/Graduation_thesis"  # 基础数据目录
        
        # 2. 加载预处理好的训练样本
        training_samples_path = base_data_dir + "/training_samples_2_6_7.npz"
        samples, labels, metadata, global_mean, global_std = load_training_samples(training_samples_path)
        
        # 3. 创建数据集
        print("\n" + "="*70)
        print("创建数据集...")
        print("="*70)
        
        try:
            full_dataset = LocalSTFTDataset(samples, labels)
        except Exception as e:
            print(f"\n[ERROR] 创建数据集失败: {e}")
            return
        
        # 4. 获取输入维度
        sample_data, _ = full_dataset[0]
        n_channels, n_time, n_freq = sample_data.shape
        print(f"\n输入维度: n_channels={n_channels}, n_time={n_time}, n_freq={n_freq}")
        
        # 5. 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n使用设备: {device}")
        
        # 6. 创建测试数据加载器
        batch_size = 32  # 最小batch size以节省内存
        test_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # 7. 加载最佳模型
        # 这里假设最佳模型文件已经存在，文件名格式为 best_dual_model_foldX.pth
        # 我们可以尝试加载所有可能的模型文件，选择F1分数最高的
        best_model = None
        best_f1 = 0
        best_model_path = None
        
        import glob
        model_files = glob.glob('best_dual_model_fold*.pth')
        # path=Path("D:/Graduation_thesis/code/")
        # model_files=path.glob("best_dual_model_fold*.pth")
        
        if not model_files:
            print("\n[ERROR] 未找到任何模型文件！")
            return
        
        print(f"\n找到 {len(model_files)} 个模型文件:")
        for model_file in model_files:
            print(f"  - {model_file}")
        
        # 尝试加载每个模型并测试
        for model_file in model_files:
            try:
                # 初始化模型
                model = RDANet(
                    in_channels=1,           
                    num_classes=2,          
                    dropout=0.3,            
                    spectral_bands=n_freq,     
                    spatial_height=n_channels,      
                    spatial_width=n_time         
                ).to(device)
                
                model.load_state_dict(torch.load(model_file))
                print(f"\n[SUCCESS] 已加载模型: {model_file}")
                
                # 测试模型
                results = test_model(model, test_loader, device)
                
                # 比较F1分数
                if results['f1'] > best_f1:
                    best_f1 = results['f1']
                    best_model = model
                    best_model_path = model_file
                    
                print(f"  F1 Score: {results['f1']:.4f}")
                
            except Exception as e:
                print(f"\n[ERROR] 加载模型 {model_file} 时出错: {e}")
                continue
        
        # 8. 输出最佳模型结果
        if best_model:
            print("\n" + "="*70)
            print(f"最佳模型: {best_model_path}")
            print(f"最佳F1 Score: {best_f1:.4f}")
            print("="*70)
            
            # 再次测试最佳模型以获取完整结果
            print("\n" + "="*70)
            print("使用最佳模型的测试结果")
            print("="*70)
            best_results = test_model(best_model, test_loader, device)
            
            # 保存测试结果
            test_results = {
                'test_data': 'training_samples_2_6_7.npz',
                'best_model': best_model_path,
                'accuracy': float(best_results['accuracy']),
                'precision': float(best_results['precision']),
                'recall': float(best_results['recall']),
                'f1': float(best_results['f1']),
                'auc': float(best_results['auc']),
                'confusion_matrix': best_results['confusion_matrix'].tolist()
            }
            
            # 保存结果到JSON文件
            with open('validation_results.json', 'w', encoding='utf-8') as f:
                json.dump(test_results, f, indent=2, ensure_ascii=False)
            
            print("\n[SUCCESS] 验证结果已保存到 validation_results.json")
            
        else:
            print("\n[ERROR] 没有成功加载任何模型！")
            
    except Exception as e:
        print(f"\n[ERROR] 验证失败: {e}")


if __name__ == "__main__":
    main()
