import json
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


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
        """加载单个STFT文件（带错误处理）"""
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
                
                # 重塑数据
                reshaped = stft_data.transpose(0, 3, 1, 2).reshape(
                    n_windows, n_time, -1
                )
                
                print(f"  重塑后: {reshaped.shape}")
                
                # 检查数据中是否有NaN或Inf
                if np.isnan(reshaped).any() or np.isinf(reshaped).any():
                    print(f"  ⚠️ 文件 {stft_file.name} 包含NaN或Inf，尝试修复")
                    reshaped = np.nan_to_num(reshaped, nan=0.0, posinf=0.0, neginf=0.0)
                
                # 存储时间信息
                self.file_original_times[file_idx] = {
                    'window_times': window_times,
                    'time_per_bin': time_per_bin,
                    'window_sec': window_sec,
                    'n_time': n_time,
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
                            'data': reshaped[w],
                            'time': mid_time,
                            'time_start': window_times[w],
                            'time_end': window_times[w] + window_sec
                        })
                else:
                    # 滑动窗口
                    for w in range(n_windows):
                        for t_start in range(0, n_time - self.window_size + 1, self.stride):
                            t_end = t_start + self.window_size
                            
                            sub_start_time = window_times[w] + t_start * time_per_bin
                            sub_end_time = window_times[w] + t_end * time_per_bin
                            sub_mid_time = (sub_start_time + sub_end_time) / 2
                            
                            window_data = reshaped[w, t_start:t_end, :]
                            
                            # 检查窗口数据
                            if window_data.size == 0:
                                continue
                                
                            samples.append({
                                'file_idx': file_idx,
                                'window_idx': w,
                                't_start': t_start,
                                't_end': t_end,
                                'data': window_data,
                                'time': sub_mid_time,
                                'time_start': sub_start_time,
                                'time_end': sub_end_time
                            })
                
                # 保存元数据
                self.file_metadata.append({
                    'file': str(stft_file),
                    'n_windows': n_windows,
                    'n_time': n_time,
                    'n_features': reshaped.shape[-1],
                    'sfreq': sfreq,
                    'window_sec': window_sec
                })
                
                return samples
                
        except Exception as e:
            print(f"  ❌ 加载文件 {stft_file.name} 时出错: {str(e)}")
            return []
    
    def _compute_global_stats(self):
        """计算全局标准化参数"""
        print("\n计算全局标准化参数...")
        
        # 如果样本太多，只使用部分样本计算
        n_samples = min(1000, len(self.samples))
        if n_samples == 0:
            print("  ⚠️ 没有样本用于计算统计量，使用默认值")
            # 使用正确的形状创建默认值
            sample_data = self.samples[0]['data']
            self.global_mean = np.zeros((1, sample_data.shape[-1]), dtype=np.float32)
            self.global_std = np.ones((1, sample_data.shape[-1]), dtype=np.float32)
            return
            
        sample_indices = np.random.choice(len(self.samples), n_samples, replace=False)
        
        # 收集数据
        all_data = []
        for idx in sample_indices:
            all_data.append(self.samples[idx]['data'])
        
        try:
            # 尝试拼接所有数据
            # 注意：这里可能会内存不足，所以可能需要分批处理
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
                            min_time = min(d.shape[0], target_shape[0])
                            min_feat = min(d.shape[1], target_shape[1])
                            d_adjusted = np.zeros(target_shape, dtype=np.float32)
                            d_adjusted[:min_time, :min_feat] = d[:min_time, :min_feat]
                            adjusted_data.append(d_adjusted)
                        else:
                            adjusted_data.append(d)
                    all_data = adjusted_data
                
                # 分批计算统计量以避免内存问题
                all_data_array = np.concatenate(all_data, axis=0)
                
                # 计算均值和标准差
                self.global_mean = np.mean(all_data_array, axis=0, keepdims=True).astype(np.float32)
                self.global_std = np.std(all_data_array, axis=0, keepdims=True).astype(np.float32) + 1e-8
                
                print(f"  均值形状: {self.global_mean.shape}")
                print(f"  标准差形状: {self.global_std.shape}")
            else:
                raise ValueError("没有有效数据")
                
        except Exception as e:
            print(f"  ⚠️ 计算统计量时出错: {e}")
            # 使用默认值
            print("  使用默认标准化参数")
            sample_data = self.samples[0]['data']
            self.global_mean = np.zeros((1, sample_data.shape[-1]), dtype=np.float32)
            self.global_std = np.ones((1, sample_data.shape[-1]), dtype=np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = sample['data'].copy()
        
        # 应用标准化
        if self.normalize and self.global_mean is not None and self.global_std is not None:
            # 确保数据维度匹配
            if data.shape[-1] != self.global_mean.shape[-1]:
                print(f"警告: 数据维度不匹配: {data.shape[-1]} vs {self.global_mean.shape[-1]}")
                # 如果维度不匹配，调整数据
                if data.shape[-1] < self.global_mean.shape[-1]:
                    # 填充
                    pad_width = ((0, 0), (0, self.global_mean.shape[-1] - data.shape[-1]))
                    data = np.pad(data, pad_width, mode='constant')
                else:
                    # 截断
                    data = data[..., :self.global_mean.shape[-1]]
            
            data = (data - self.global_mean) / self.global_std
        
        # 确保数据是float32类型
        data = data.astype(np.float32)
        
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
        
        # 生成标签
        print("\n生成标签...")
        self.labels = []
        self.label_info = []
        
        seizure_count = 0
        non_seizure_count = 0
        
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
            
            # 判断是否为发作
            is_seizure = 0
            matched_time = None
            
            if file_name in seizure_times:
                for start, end in seizure_times[file_name]:
                    # 如果窗口与发作区间重叠，标记为发作
                    if (time_start <= end and time_end >= start):
                        is_seizure = 1
                        matched_time = (start, end)
                        break
            
            self.labels.append(is_seizure)
            self.label_info.append({
                'file_idx': file_idx,
                'file_name': file_name,
                'time_range': (time_start, time_end),
                'label': is_seizure,
                'matched_seizure': matched_time
            })
            
            if is_seizure == 1:
                seizure_count += 1
            else:
                non_seizure_count += 1
        
        self.labels = np.array(self.labels)
        
        # 统计
        print(f"\n类别分布:")
        print(f"  类别0 (非发作): {non_seizure_count} ({non_seizure_count/len(self.labels)*100:.2f}%)")
        print(f"  类别1 (发作): {seizure_count} ({seizure_count/len(self.labels)*100:.2f}%)")
        
        # 显示发作样本示例
        if seizure_count > 0:
            print(f"\n发作样本示例:")
            seizure_indices = np.where(self.labels == 1)[0]
            for i in seizure_indices[:min(10, len(seizure_indices))]:
                info = self.label_info[i]
                print(f"  样本 {i}: 文件 {info['file_name']}, "
                      f"时间范围 {info['time_range'][0]:.1f}-{info['time_range'][1]:.1f}秒, "
                      f"匹配发作 {info['matched_seizure']}")
        else:
            print("\n⚠️ 警告: 没有找到任何发作样本！")
            print("请检查以下文件是否应该有发作:")
            for file_name in seizure_times.keys():
                print(f"  - {file_name}")
        
        # 平衡类别
        self.balance_classes = balance_classes
        if balance_classes and seizure_count > 0 and non_seizure_count > 0:
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

class RDANet(nn.Module):
    """
    RDANet模型，用于脑电信号频谱图分类
    """
    def __init__(self, in_channels=16, num_classes=2, dropout=0.3):
        super().__init__()
        
        # 初始卷积层 - 不使用池化，避免维度过小
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 调整卷积
        self.adjust_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 四个残差块
        self.residual_blocks = nn.Sequential(
            ResidualBlock(64, 64, dropout),
            ResidualBlock(64, 64, dropout),
            ResidualBlock(64, 64, dropout),
            ResidualBlock(64, 64, dropout)
        )
        
        # 注意力模块
        self.channel_attention = ChannelAttention(64)
        self.spatial_attention = SpatialAttention()
        
        # 全局平均池化和分类器
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # 输入形状: (batch, time, features)
        # 转换为: (batch, features, time) -> (batch, features, time, 1) -> (batch, features, 1, time)
        x = x.permute(0, 2, 1).unsqueeze(2)
        
        # 初始卷积
        x = self.initial_conv(x)
        
        # 调整卷积
        x = self.adjust_conv(x)
        
        # 残差块
        x = self.residual_blocks(x)
        
        # 通道注意力
        x = self.channel_attention(x)
        
        # 空间注意力
        x = self.spatial_attention(x)
        
        # 全局平均池化
        x = self.global_avg_pool(x)
        
        # 分类
        output = self.classifier(x)
        
        return output


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

# ========== 2. 训练函数 ==========

def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='cuda', fold=None):
    """
    训练模型
    """
    # 损失函数和优化器
    class_weights = torch.FloatTensor([1.0, 5.0]).to(device)  # 给发作类更高权重
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        train_batches = 0
        
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        history['train_loss'].append(avg_train_loss)
        
        # 验证
        model.eval()
        val_loss = 0
        val_batches = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_batches += 1
                
                _, predicted = torch.max(outputs, 1)
                _, true = torch.max(labels, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(true.cpu().numpy())
        
        avg_val_loss = val_loss / val_batches
        history['val_loss'].append(avg_val_loss)
        
        # 计算验证指标
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        
        history['val_accuracy'].append(accuracy)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)
        history['val_f1'].append(f1)
        
        # 学习率调整
        scheduler.step(avg_val_loss)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # 根据是否为K折交叉验证选择不同的文件名
            if fold is not None:
                model_path = f'best_rdanet_model_fold{fold+1}.pth'
            else:
                model_path = 'best_rdanet_model.pth'
            torch.save(model.state_dict(), model_path)
        
        # 打印进度
        if (epoch + 1) % 5 == 0:
            fold_info = f" Fold {fold+1}" if fold is not None else ""
            print(f"\nEpoch {epoch+1}/{epochs}{fold_info}")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            print(f"  Val Accuracy: {accuracy:.4f}")
            print(f"  Val Precision: {precision:.4f}")
            print(f"  Val Recall: {recall:.4f}")
            print(f"  Val F1: {f1:.4f}")
            
            # 打印混淆矩阵
            if epoch == epochs - 1 or epoch == 4:
                cm = confusion_matrix(all_labels, all_preds)
                print("\n混淆矩阵:")
                print(f"         预测非发作  预测发作")
                print(f"实际非发作:   {cm[0,0]:5d}      {cm[0,1]:5d}")
                print(f"实际发作:     {cm[1,0]:5d}      {cm[1,1]:5d}")
    
    return history


# ========== 3. 绘制训练曲线 ==========

def plot_kfold_training_history(all_fold_histories, n_folds):
    """
    绘制K折交叉验证的训练曲线
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # 颜色列表
    colors = plt.cm.tab10(np.linspace(0, 1, n_folds))
    
    # 1. 损失曲线
    for fold_idx, history in enumerate(all_fold_histories):
        axes[0, 0].plot(history['train_loss'], label=f'Fold {fold_idx+1} Train', 
                       color=colors[fold_idx], linestyle='-', alpha=0.7)
        axes[0, 0].plot(history['val_loss'], label=f'Fold {fold_idx+1} Val', 
                       color=colors[fold_idx], linestyle='--', alpha=0.7)
    
    # 计算平均损失曲线
    max_epochs = max(len(h['train_loss']) for h in all_fold_histories)
    avg_train_loss = []
    avg_val_loss = []
    std_train_loss = []
    std_val_loss = []
    
    for epoch in range(max_epochs):
        train_losses = [h['train_loss'][epoch] for h in all_fold_histories if epoch < len(h['train_loss'])]
        val_losses = [h['val_loss'][epoch] for h in all_fold_histories if epoch < len(h['val_loss'])]
        
        avg_train_loss.append(np.mean(train_losses))
        avg_val_loss.append(np.mean(val_losses))
        std_train_loss.append(np.std(train_losses))
        std_val_loss.append(np.std(val_losses))
    
    epochs = range(1, max_epochs + 1)
    axes[0, 0].plot(epochs, avg_train_loss, label='Average Train', color='black', 
                   linewidth=2, linestyle='-')
    axes[0, 0].plot(epochs, avg_val_loss, label='Average Val', color='black', 
                   linewidth=2, linestyle='--')
    
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title(f'Training and Validation Loss ({n_folds}-Fold CV)')
    axes[0, 0].legend(loc='best', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 准确率曲线
    for fold_idx, history in enumerate(all_fold_histories):
        axes[0, 1].plot(history['val_accuracy'], label=f'Fold {fold_idx+1}', 
                       color=colors[fold_idx], alpha=0.7)
    
    avg_accuracy = []
    std_accuracy = []
    for epoch in range(max_epochs):
        accuracies = [h['val_accuracy'][epoch] for h in all_fold_histories if epoch < len(h['val_accuracy'])]
        avg_accuracy.append(np.mean(accuracies))
        std_accuracy.append(np.std(accuracies))
    
    axes[0, 1].plot(epochs, avg_accuracy, label='Average', color='black', linewidth=2)
    axes[0, 1].fill_between(epochs, 
                           np.array(avg_accuracy) - np.array(std_accuracy),
                           np.array(avg_accuracy) + np.array(std_accuracy),
                           alpha=0.2, color='gray')
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title(f'Validation Accuracy ({n_folds}-Fold CV)')
    axes[0, 1].legend(loc='best', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Precision/Recall/F1曲线
    for fold_idx, history in enumerate(all_fold_histories):
        axes[1, 0].plot(history['val_precision'], label=f'Fold {fold_idx+1} Precision', 
                       color=colors[fold_idx], linestyle='-', alpha=0.5)
        axes[1, 0].plot(history['val_recall'], label=f'Fold {fold_idx+1} Recall', 
                       color=colors[fold_idx], linestyle='--', alpha=0.5)
        axes[1, 0].plot(history['val_f1'], label=f'Fold {fold_idx+1} F1', 
                       color=colors[fold_idx], linestyle=':', alpha=0.5)
    
    # 计算平均指标
    avg_precision = []
    avg_recall = []
    avg_f1 = []
    
    for epoch in range(max_epochs):
        precisions = [h['val_precision'][epoch] for h in all_fold_histories if epoch < len(h['val_precision'])]
        recalls = [h['val_recall'][epoch] for h in all_fold_histories if epoch < len(h['val_recall'])]
        f1s = [h['val_f1'][epoch] for h in all_fold_histories if epoch < len(h['val_f1'])]
        
        avg_precision.append(np.mean(precisions))
        avg_recall.append(np.mean(recalls))
        avg_f1.append(np.mean(f1s))
    
    axes[1, 0].plot(epochs, avg_precision, label='Average Precision', color='red', linewidth=2)
    axes[1, 0].plot(epochs, avg_recall, label='Average Recall', color='blue', linewidth=2)
    axes[1, 0].plot(epochs, avg_f1, label='Average F1', color='green', linewidth=2)
    
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title(f'Precision, Recall, and F1 Score ({n_folds}-Fold CV)')
    axes[1, 0].legend(loc='best', fontsize=7)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 最终指标表格和箱线图
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    
    # 收集所有折的最终指标
    final_metrics = {
        'Accuracy': [h['val_accuracy'][-1] for h in all_fold_histories],
        'Precision': [h['val_precision'][-1] for h in all_fold_histories],
        'Recall': [h['val_recall'][-1] for h in all_fold_histories],
        'F1': [h['val_f1'][-1] for h in all_fold_histories]
    }
    
    # 创建表格
    table_data = [
        ['Metric', 'Mean', 'Std', 'Min', 'Max']
    ]
    for metric_name, values in final_metrics.items():
        table_data.append([
            metric_name,
            f"{np.mean(values):.4f}",
            f"{np.std(values):.4f}",
            f"{np.min(values):.4f}",
            f"{np.max(values):.4f}"
        ])
    
    table = axes[1, 1].table(
        cellText=table_data,
        loc='center',
        cellLoc='center',
        colWidths=[0.2, 0.2, 0.2, 0.2, 0.2]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    # 设置表头样式
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle(f'DCRNN K-Fold Cross-Validation Training History ({n_folds} Folds)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('kfold_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 额外创建一个箱线图来展示各折的性能分布
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1']
    data_to_plot = [final_metrics[m] for m in metrics_to_plot]
    
    bp = ax.boxplot(data_to_plot, labels=metrics_to_plot, patch_artist=True)
    
    # 设置箱线图颜色
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Score')
    ax.set_title(f'Performance Distribution Across {n_folds} Folds')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加均值线
    for i, data in enumerate(data_to_plot):
        mean_val = np.mean(data)
        ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.5, xmin=i/len(metrics_to_plot)+0.05, 
                  xmax=(i+1)/len(metrics_to_plot)-0.05)
    
    plt.tight_layout()
    plt.savefig('kfold_performance_boxplot.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_training_history(history):
    """
    绘制训练曲线（单次训练）
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失曲线
    axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', color='orange')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[0, 1].plot(history['val_accuracy'], label='Val Accuracy', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision/Recall/F1曲线
    axes[1, 0].plot(history['val_precision'], label='Precision', color='red')
    axes[1, 0].plot(history['val_recall'], label='Recall', color='blue')
    axes[1, 0].plot(history['val_f1'], label='F1', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Precision, Recall, and F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 最终指标表格
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    
    final_metrics = [
        ['Metric', 'Value'],
        ['Best Val Loss', f"{min(history['val_loss']):.4f}"],
        ['Final Accuracy', f"{history['val_accuracy'][-1]:.4f}"],
        ['Final Precision', f"{history['val_precision'][-1]:.4f}"],
        ['Final Recall', f"{history['val_recall'][-1]:.4f}"],
        ['Final F1', f"{history['val_f1'][-1]:.4f}"]
    ]
    
    table = axes[1, 1].table(
        cellText=final_metrics,
        loc='center',
        cellLoc='center',
        colWidths=[0.3, 0.3]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.suptitle('DCRNN Training History on chb STFT Data (Patients 01-03)', fontsize=14)
    plt.tight_layout()
    plt.savefig('dcrnn_training_history_multi_patient.png', dpi=150)
    plt.show()


# ========== 4. 测试函数 ==========

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
    
    # 计算AUC
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(all_labels, all_probs)
    
    print("\n" + "="*60)
    print("测试结果")
    print("="*60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print("\n混淆矩阵:")
    print(f"         预测非发作  预测发作")
    print(f"实际非发作:   {cm[0,0]:5d}      {cm[0,1]:5d}")
    print(f"实际发作:     {cm[1,0]:5d}      {cm[1,1]:5d}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm
    }


# ========== 5. 主程序 ==========

def main():
    print("\n" + "="*70)
    print("开始训练RDANet模型 on chb01-03 STFT数据 (K折交叉验证)")
    print("="*70)
    
    # 1. 设置基础路径
    base_data_dir = "D:/Graduation_thesis"  # 基础数据目录
    
    # 2. 指定要加载的病人
    patient_ids = ['chb01', 'chb02']
    
    # 3. 设置K折交叉验证参数
    n_folds = 5  # K折数
    print(f"\n使用 {n_folds} 折交叉验证")
    
    # 4. 加载所有病人的数据
    all_stft_files, all_seizure_times = load_all_patients_data(
        patient_ids=patient_ids,
        base_data_dir=base_data_dir
    )
    
    if len(all_stft_files) == 0:
        print("\n❌ 错误: 没有找到任何STFT文件！")
        return
    
    # 5. 创建数据集
    print("\n" + "="*70)
    print("创建数据集...")
    print("="*70)
    
    try:
        full_dataset = MultiFileSTFTWithLabels(
            stft_files=all_stft_files,
            seizure_times=all_seizure_times,
            window_size=30,
            stride=15,
            normalize=True,
            balance_classes=True
        )
    except Exception as e:
        print(f"\n❌ 创建数据集失败: {e}")
        return
    
    # 6. 获取输入维度
    sample_data, _ = full_dataset[0]
    time_steps, features = sample_data.shape
    print(f"\n输入维度: time_steps={time_steps}, features={features}")
    
    # 7. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 8. K折交叉验证
    print("\n" + "="*70)
    print(f"开始 {n_folds} 折交叉验证")
    print("="*70)
    
    # 创建KFold对象
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # 存储所有折的结果
    all_fold_results = []
    all_fold_histories = []
    
    # 遍历每一折
    for fold, (train_idx, test_idx) in enumerate(kfold.split(range(len(full_dataset)))):
        print(f"\n{'='*70}")
        print(f"第 {fold+1}/{n_folds} 折")
        print(f"{'='*70}")
        
        # 创建训练集和测试集
        train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        test_dataset = torch.utils.data.Subset(full_dataset, test_idx)
        
        # 从训练集中划分验证集
        train_size = int(0.85 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        
        # 创建数据加载器
        batch_size = min(16, len(train_dataset))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # 初始化模型
        model = RDANet(
            in_channels=features,
            num_classes=2,
            dropout=0.3
        ).to(device)
        
        print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
        
        # 训练模型
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=50,
            lr=0.001,
            device=device,
            fold=fold
        )
        
        # 加载最佳模型并测试
        model_path = f'best_rdanet_model_fold{fold+1}.pth'
        if Path(model_path).exists():
            model.load_state_dict(torch.load(model_path))
            test_results = test_model(model, test_loader, device)
        else:
            print(f"\n⚠️ 未找到最佳模型文件 {model_path}，使用最后一个epoch的模型进行测试")
            test_results = test_model(model, test_loader, device)
        
        # 保存当前折的结果
        all_fold_results.append(test_results)
        all_fold_histories.append(history)
        
        print(f"\n✅ 第 {fold+1} 折完成！")
    
    # 9. 汇总所有折的结果
    print("\n" + "="*70)
    print("K折交叉验证结果汇总")
    print("="*70)
    
    # 计算平均指标
    avg_accuracy = np.mean([r['accuracy'] for r in all_fold_results])
    avg_precision = np.mean([r['precision'] for r in all_fold_results])
    avg_recall = np.mean([r['recall'] for r in all_fold_results])
    avg_f1 = np.mean([r['f1'] for r in all_fold_results])
    avg_auc = np.mean([r['auc'] for r in all_fold_results])
    
    std_accuracy = np.std([r['accuracy'] for r in all_fold_results])
    std_precision = np.std([r['precision'] for r in all_fold_results])
    std_recall = np.std([r['recall'] for r in all_fold_results])
    std_f1 = np.std([r['f1'] for r in all_fold_results])
    std_auc = np.std([r['auc'] for r in all_fold_results])
    
    print(f"\n平均指标:")
    print(f"  Accuracy:  {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"  Precision: {avg_precision:.4f} ± {std_precision:.4f}")
    print(f"  Recall:    {avg_recall:.4f} ± {std_recall:.4f}")
    print(f"  F1 Score:  {avg_f1:.4f} ± {std_f1:.4f}")
    print(f"  AUC:       {avg_auc:.4f} ± {std_auc:.4f}")
    
    # 12. 选择最佳模型
    print("\n" + "="*70)
    print("选择最佳模型")
    print("="*70)
    
    # 使用F1分数作为主要指标选择最佳模型
    best_fold_idx = np.argmax([r['f1'] for r in all_fold_results])
    best_fold = best_fold_idx + 1
    best_results = all_fold_results[best_fold_idx]
    
    print(f"\n最佳模型: 第 {best_fold} 折")
    print(f"  F1 Score:  {best_results['f1']:.4f}")
    print(f"  Accuracy:  {best_results['accuracy']:.4f}")
    print(f"  Precision: {best_results['precision']:.4f}")
    print(f"  Recall:    {best_results['recall']:.4f}")
    print(f"  AUC:       {best_results['auc']:.4f}")
    
    # # 13. 在chb03上测试最佳模型
    # print("\n" + "="*70)
    # print("在chb03上测试最佳模型")
    # print("="*70)
    
    # test_patient_ids = ['chb03']
    # test_stft_files, test_seizure_times = load_all_patients_data(
    #     patient_ids=test_patient_ids,
    #     base_data_dir=base_data_dir
    # )
    
    # if len(test_stft_files) == 0:
    #     print("\n❌ 错误: 没有找到chb03的STFT文件！")
    # else:
    #     # 创建测试数据集
    #     try:
    #         test_dataset = MultiFileSTFTWithLabels(
    #             stft_files=test_stft_files,
    #             seizure_times=test_seizure_times,
    #             window_size=30,
    #             stride=15,
    #             normalize=True,
    #             balance_classes=False  # 测试集不需要平衡
    #         )
            
    #         print(f"\nchb03测试集大小: {len(test_dataset)}")
            
    #         # 创建测试数据加载器
    #         test_batch_size = min(16, len(test_dataset))
    #         test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0)
            
    #         # 加载最佳模型
    #         best_model_path = f'best_dcrnn_model_fold{best_fold}.pth'
    #         if Path(best_model_path).exists():
    #             # 重新初始化模型
    #             best_model = DCRNNForSTFT(
    #                 input_size=features,
    #                 hidden_size=128,
    #                 num_layers=2,
    #                 num_classes=2,
    #                 dropout=0.3
    #             ).to(device)
                
    #             best_model.load_state_dict(torch.load(best_model_path))
    #             print(f"\n✅ 已加载最佳模型: {best_model_path}")
                
        #         # 在chb03上测试
        #         chb03_results = test_model(best_model, test_loader, device)
                
        #         # 保存chb03测试结果
        #         chb03_converted = {
        #             'test_patient': 'chb03',
        #             'best_fold': best_fold,
        #             'accuracy': float(chb03_results['accuracy']),
        #             'precision': float(chb03_results['precision']),
        #             'recall': float(chb03_results['recall']),
        #             'f1': float(chb03_results['f1']),
        #             'auc': float(chb03_results['auc']),
        #             'confusion_matrix': chb03_results['confusion_matrix'].tolist()
        #         }
                
        #         summary_results['chb03_test_results'] = chb03_converted
                
        #         print(f"\n✅ chb03测试完成！")
                
        #     else:
        #         print(f"\n❌ 未找到最佳模型文件: {best_model_path}")
                
        # except Exception as e:
        #     print(f"\n❌ chb03测试失败: {e}")
    
    # 10. 绘制所有折的训练曲线
    plot_kfold_training_history(all_fold_histories, n_folds)
    
    # 11. 保存结果
    converted_results = []
    for i, (results, history) in enumerate(zip(all_fold_results, all_fold_histories)):
        converted_history = {}
        for key, values in history.items():
            converted_history[key] = [float(v) for v in values]
        
        converted_result = {
            'fold': i + 1,
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1': float(results['f1']),
            'auc': float(results['auc']),
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'history': converted_history
        }
        converted_results.append(converted_result)
    
    # 保存汇总结果
    summary_results = {
        'patients': patient_ids,
        'num_stft_files': len(all_stft_files),
        'num_seizure_files': len(all_seizure_times),
        'n_folds': n_folds,
        'fold_results': converted_results,
        'average_metrics': {
            'accuracy': {'mean': float(avg_accuracy), 'std': float(std_accuracy)},
            'precision': {'mean': float(avg_precision), 'std': float(std_precision)},
            'recall': {'mean': float(avg_recall), 'std': float(std_recall)},
            'f1': {'mean': float(avg_f1), 'std': float(std_f1)},
            'auc': {'mean': float(avg_auc), 'std': float(std_auc)}
        }
    }
    
    with open('kfold_training_results.json', 'w') as f:
        json.dump(summary_results, f, indent=2)
    
    print("\n✅ K折交叉验证完成！结果已保存到 kfold_training_results.json")
    
    # # 14. 打印chb03测试结果摘要
    # if 'chb03_test_results' in summary_results:
    #     print("\n" + "="*70)
    #     print("chb03测试结果摘要")
    #     print("="*70)
    #     chb03 = summary_results['chb03_test_results']
    #     print(f"\n测试病人: {chb03['test_patient']}")
    #     print(f"使用模型: 第 {chb03['best_fold']} 折")
    #     print(f"\n性能指标:")
    #     print(f"  Accuracy:  {chb03['accuracy']:.4f}")
    #     print(f"  Precision: {chb03['precision']:.4f}")
    #     print(f"  Recall:    {chb03['recall']:.4f}")
    #     print(f"  F1 Score:  {chb03['f1']:.4f}")
    #     print(f"  AUC:       {chb03['auc']:.4f}")
    #     print(f"\n混淆矩阵:")
    #     cm = np.array(chb03['confusion_matrix'])
    #     print(f"         预测非发作  预测发作")
    #     print(f"实际非发作:   {cm[0,0]:5d}      {cm[0,1]:5d}")
    #     print(f"实际发作:     {cm[1,0]:5d}      {cm[1,1]:5d}")


if __name__ == "__main__":
    main()