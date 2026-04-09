#分别加载数据集
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


# def parse_chb_summary(txt_file_path):
#     """
#     从summary.txt文件解析发作时间
#     """
#     print(f"\n从文件加载发作时间: {txt_file_path}")
#     print("="*60)
    
#     try:
#         with open(txt_file_path, 'r', encoding='utf-8') as f:
#             content = f.read()
#     except Exception as e:
#         print(f"❌ 读取文件失败: {e}")
#         return {}, 256
    
#     # 解析采样率
#     rate_match = re.search(r'Data Sampling Rate:\s*(\d+)\s*Hz', content)
#     sampling_rate = int(rate_match.group(1)) if rate_match else 256
#     print(f"采样率: {sampling_rate} Hz")
    
#     # 解析发作时间
#     seizure_times = {}
    
#     # 按文件分割
#     file_pattern = r'File Name:\s*(.+?\.edf).*?Number of Seizures in File:\s*(\d+)(.*?)(?=File Name:|\Z)'
#     file_matches = re.findall(file_pattern, content, re.DOTALL | re.IGNORECASE)
    
#     print(f"找到 {len(file_matches)} 个文件记录")
    
#     for file_name, n_seizures_str, file_content in file_matches:
#         file_name = file_name.strip()
#         n_seizures = int(n_seizures_str)
        
#         if n_seizures > 0:
#             # 解析发作时间
#             seizure_pattern = r'Seizure Start Time:\s*(\d+)\s*seconds.*?Seizure End Time:\s*(\d+)\s*seconds'
#             seizures = re.findall(seizure_pattern, file_content, re.DOTALL | re.IGNORECASE)
            
#             times = []
#             for start, end in seizures:
#                 start_time = int(start)
#                 end_time = int(end)
#                 times.append((start_time, end_time))
#                 print(f"  {file_name}: {start_time}-{end_time}秒")
            
#             seizure_times[file_name] = times
    
#     print(f"\n总共找到 {len(seizure_times)} 个包含发作的文件")
#     return seizure_times, sampling_rate


# def load_all_patients_data(patient_ids, base_data_dir, stft_subdir="stft_data"):
#     """
#     加载多个病人的STFT数据和发作时间（带错误处理）
#     """
#     all_stft_files = []
#     all_seizure_times = {}
#     failed_patients = []
    
#     print("\n" + "="*70)
#     print(f"加载病人数据: {patient_ids}")
#     print("="*70)
    
#     for patient_id in patient_ids:
#         print(f"\n--- 处理病人 {patient_id} ---")
        
#         # STFT文件路径
#         stft_dir = Path(base_data_dir) / "code" / f"{stft_subdir}/{patient_id}"
#         if not stft_dir.exists():
#             print(f"⚠️ 警告: STFT目录不存在: {stft_dir}")
#             failed_patients.append(patient_id)
#             continue
            
#         stft_files = sorted(stft_dir.glob("*_stft_30s.npz"))
#         print(f"找到 {len(stft_files)} 个STFT文件")
        
#         # 发作时间文件路径
#         summary_file = Path(base_data_dir) / f"data/chb-mit-scalp-eeg-database-1.0.0/{patient_id}/{patient_id}-summary.txt"
#         if not summary_file.exists():
#             print(f"⚠️ 警告: summary文件不存在: {summary_file}")
#             failed_patients.append(patient_id)
#             continue
        
#         try:    
#             seizure_times, _ = parse_chb_summary(summary_file)
            
#             # 添加到总列表
#             all_stft_files.extend(stft_files)
#             all_seizure_times.update(seizure_times)
            
#             print(f"病人 {patient_id} 发作文件数: {len(seizure_times)}")
#         except Exception as e:
#             print(f"⚠️ 处理病人 {patient_id} 时出错: {e}")
#             failed_patients.append(patient_id)
#             continue
    
#     print(f"\n总计:")
#     print(f"  STFT文件总数: {len(all_stft_files)}")
#     print(f"  发作文件总数: {len(all_seizure_times)}")
#     if failed_patients:
#         print(f"  处理失败的病人: {failed_patients}")
    
#     return all_stft_files, all_seizure_times

# class RDANet(nn.Module):
#     """
#     RDANet模型，用于脑电信号频谱图分类 - 修改版：输入为(batch, 22, n_time, n_freq)
#     """
#     def __init__(self, in_channels=22, num_classes=2, dropout=0.3):
#         super().__init__()
        
#         # 初始卷积层 - 输入通道数为in_channels (22)
#         self.initial_conv = nn.Sequential(
#             nn.Conv2d(in_channels, 32, kernel_size=(3, 3), padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
        
#         # 调整卷积
#         self.adjust_conv = nn.Sequential(
#             nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
        
#         # 两个残差块
#         self.residual_blocks = nn.Sequential(
#             ResidualBlock(32, 32, dropout),
#             ResidualBlock(32, 32, dropout)
#         )
        
#         # 注意力模块
#         self.channel_attention = ChannelAttention(32)
#         self.spatial_attention = SpatialAttention()
        
#         # 全局平均池化和分类器
#         self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(32, 64),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(64, num_classes)
#         )
    
#     def forward(self, x):
#         # 输入形状: (batch, n_channels, n_time, n_freq) = (batch, 22, 59, ~120)
        
#         # 初始卷积 - 直接处理4D输入
#         x = self.initial_conv(x)  # 输出: (batch, 32, n_time, n_freq)
        
#         # 调整卷积
#         x = self.adjust_conv(x)  # 输出: (batch, 32, n_time, n_freq)
        
#         # 残差块
#         x = self.residual_blocks(x)  # 输出: (batch, 32, n_time, n_freq)
        
#         # 通道注意力与空间注意力的组合
#         x1 = self.channel_attention(x)
#         x2 = self.spatial_attention(x)
#         x = x1 + x2
        
#         # 全局平均池化
#         x = self.global_avg_pool(x)  # 输出: (batch, 32, 1, 1)
        
#         # 分类
#         output = self.classifier(x)  # 输出: (batch, num_classes)
        
#         return output


# class ResidualBlock(nn.Module):
#     """
#     残差块
#     """
#     def __init__(self, in_channels, out_channels, dropout=0.3):
#         super().__init__()
        
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)
        
#         #  shortcut连接
#         if in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
#                 nn.BatchNorm2d(out_channels)
#             )
#         else:
#             self.shortcut = nn.Identity()
    
#     def forward(self, x):
#         residual = self.shortcut(x)
        
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
        
#         x = self.conv2(x)
#         x = self.bn2(x)
        
#         x += residual
#         x = self.relu(x)
        
#         return x


# class ChannelAttention(nn.Module):
#     """
#     通道注意力模块
#     """
#     def __init__(self, in_channels, reduction_ratio=16):
#         super().__init__()
        
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
#         self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, x):
#         batch_size, channels, height, width = x.shape
        
#         # 平均池化路径
#         avg_out = self.avg_pool(x).view(batch_size, channels)
#         avg_out = self.fc1(avg_out)
#         avg_out = self.relu(avg_out)
#         avg_out = self.fc2(avg_out)
        
#         # 最大池化路径
#         max_out = self.max_pool(x).view(batch_size, channels)
#         max_out = self.fc1(max_out)
#         max_out = self.relu(max_out)
#         max_out = self.fc2(max_out)
        
#         # 融合
#         out = avg_out + max_out
#         out = self.sigmoid(out).view(batch_size, channels, 1, 1)
        
#         return x * out


# class SpatialAttention(nn.Module):
#     """
#     空间注意力模块
#     """
#     def __init__(self, kernel_size=7):
#         super().__init__()
        
#         self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, x):
#         # 通道维度的平均池化
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         # 通道维度的最大池化
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         # 拼接
#         out = torch.cat([avg_out, max_out], dim=1)
#         # 卷积
#         out = self.conv(out)
#         # 激活
#         out = self.sigmoid(out)
        
#         return x * out

# ========== 2. 训练函数 ==========

def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='cuda', fold=None, patience=10):
    """
    训练模型
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 最大训练轮数
        lr: 学习率
        device: 设备
        fold: 折数（用于K折交叉验证）
        patience: Early stopping的耐心值，验证损失连续patience个epoch不改善则停止
    """
    # 损失函数和优化器
    class_weights = torch.FloatTensor([1.0, 2.0]).to(device)  # 给发作类更高权重
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
    
    # Early stopping相关变量
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    early_stop = False
    
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
        
        # Early stopping检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
            
            # 保存最佳模型到文件
            if fold is not None:
                model_path = f'best_dual_model_fold{fold}.pth'
            else:
                model_path = 'best_dual_model.pth'
            torch.save(model.state_dict(), model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                early_stop = True
                print(f"\nEarly stopping triggered! 验证损失连续{patience}个epoch未改善。")
                print(f"最佳验证损失: {best_val_loss:.4f} 在 epoch {epoch - patience + 1}")
                break
        
        # 打印进度
        if (epoch + 1) % 5 == 0 or early_stop:
            fold_info = f" Fold {fold}" if fold is not None else ""
            print(f"\nEpoch {epoch+1}/{epochs}{fold_info}")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            print(f"  Val Accuracy: {accuracy:.4f}")
            print(f"  Val Precision: {precision:.4f}")
            print(f"  Val Recall: {recall:.4f}")
            print(f"  Val F1: {f1:.4f}")
            
            # 打印混淆矩阵
            if epoch == epochs - 1 or epoch == 4 or early_stop:
                cm = confusion_matrix(all_labels, all_preds)
                print("\n混淆矩阵:")
                print(f"         预测非发作  预测发作")
                print(f"实际非发作:   {cm[0,0]:5d}      {cm[0,1]:5d}")
                print(f"实际发作:     {cm[1,0]:5d}      {cm[1,1]:5d}")
    
    # 训练结束后的信息
    if early_stop:
        print(f"\n训练提前结束，实际训练了 {epoch + 1} 个epoch")
        print(f"最佳验证损失: {best_val_loss:.4f}")
    else:
        print(f"\n训练完成，共训练了 {epochs} 个epoch")
        print(f"最佳验证损失: {best_val_loss:.4f}")
    
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


# ========== 5. 主程序 ==========

def main():
    print("\n" + "="*70)
    print("开始训练RDANet模型 on STFT数据 (直接加载训练/验证/测试集)")
    print("="*70)
    
    # 1. 设置基础路径
    base_data_dir = "D:/Graduation_thesis"  # 基础数据目录
    
    # 2. 加载训练、验证、测试样本
    print("\n" + "="*70)
    print("加载数据集...")
    print("="*70)
    
    train_path = base_data_dir + "/training_samples_2_3_train.npz"
    val_path = base_data_dir + "/training_samples_2_3_val.npz"
    test_path = base_data_dir + "/training_samples_2_3_test.npz"
    
    try:
        # 加载训练集
        print(f"\n加载训练集: {train_path}")
        train_samples, train_labels, train_metadata, train_mean, train_std = load_training_samples(train_path)
        train_dataset = LocalSTFTDataset(train_samples, train_labels)
        
        # 加载验证集
        print(f"\n加载验证集: {val_path}")
        val_samples, val_labels, val_metadata, val_mean, val_std = load_training_samples(val_path)
        val_dataset = LocalSTFTDataset(val_samples, val_labels)
        
        # 加载测试集
        print(f"\n加载测试集: {test_path}")
        test_samples, test_labels, test_metadata, test_mean, test_std = load_training_samples(test_path)
        test_dataset = LocalSTFTDataset(test_samples, test_labels)
        
    except Exception as e:
        print(f"\n❌ 加载数据集失败: {e}")
        return
    
    # 3. 获取输入维度
    sample_data, _ = train_dataset[0]
    n_channels, n_time, n_freq = sample_data.shape
    print(f"\n输入维度: n_channels={n_channels}, n_time={n_time}, n_freq={n_freq}")
    
    print(f"\n数据集大小:")
    print(f"  训练集: {len(train_dataset)}")
    print(f"  验证集: {len(val_dataset)}")
    print(f"  测试集: {len(test_dataset)}")
    
    # 4. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 5. 创建数据加载器
    print("\n" + "="*70)
    print("创建数据加载器...")
    print("="*70)
    
    batch_size = 16  # batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 6. 初始化模型
    print("\n" + "="*70)
    print("初始化模型...")
    print("="*70)
    
    model = RDANet(
        in_channels=1,           
        num_classes=2,          
        dropout=0.5,            
        spectral_bands=n_freq,     
        spatial_height=n_channels,      
        spatial_width=n_time         
    ).to(device)
    
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 7. 训练模型
    print("\n" + "="*70)
    print("开始训练模型")
    print("="*70)
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        lr=0.0005,
        device=device,
        fold=0,  # 使用fold=0表示非K折训练
        patience=10  # Early stopping耐心值：验证损失连续10个epoch不改善则停止
    )
    
    # 8. 加载最佳模型并测试
    print("\n" + "="*70)
    print("测试模型")
    print("="*70)
    
    model_path = 'best_dual_model_fold0.pth'
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path))
        print(f"✅ 加载最佳模型: {model_path}")
    else:
        print(f"⚠️ 未找到最佳模型文件 {model_path}，使用最后一个epoch的模型进行测试")
    
    test_results = test_model(model, test_loader, device)
    
    # 9. 打印测试结果
    print("\n" + "="*70)
    print("测试结果")
    print("="*70)
    print(f"  Accuracy:    {test_results['accuracy']:.4f}")
    print(f"  Precision:   {test_results['precision']:.4f}")
    print(f"  Recall:      {test_results['recall']:.4f}")
    print(f"  F1 Score:    {test_results['f1']:.4f}")
    print(f"  Sensitivity: {test_results['sensitivity']:.4f}  (发作样本正确识别率)")
    print(f"  Specificity: {test_results['specificity']:.4f}  (非发作样本正确识别率)")
    print(f"  AUC:         {test_results['auc']:.4f}")
    
    # 10. 绘制训练曲线
    print("\n" + "="*70)
    print("绘制训练曲线...")
    print("="*70)
    plot_training_history(history)
    
    # 11. 保存结果
    print("\n" + "="*70)
    print("保存训练结果...")
    print("="*70)
    
    converted_history = {}
    for key, values in history.items():
        converted_history[key] = [float(v) for v in values]
    
    results = {
        'test_results': {
            'accuracy': float(test_results['accuracy']),
            'precision': float(test_results['precision']),
            'recall': float(test_results['recall']),
            'f1': float(test_results['f1']),
            'auc': float(test_results['auc']),
            'sensitivity': float(test_results['sensitivity']),
            'specificity': float(test_results['specificity']),
            'confusion_matrix': test_results['confusion_matrix'].tolist()
        },
        'history': converted_history,
        'model_config': {
            'in_channels': 1,
            'num_classes': 2,
            'dropout': 0.5,
            'spectral_bands': n_freq,
            'spatial_height': n_channels,
            'spatial_width': n_time
        },
        'dataset_info': {
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'test_size': len(test_dataset),
            'n_channels': n_channels,
            'n_time': n_time,
            'n_freq': n_freq
        }
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ 训练完成！结果已保存到 training_results.json")
    print(f"\n最佳模型已保存: {model_path}")
    






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
        #self.position_query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        # self.position_key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        # self.position_value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.position_query = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.position_key = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.position_value = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
       
        
        # 通道注意力
        # self.channel_query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # self.channel_key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # self.channel_value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # 输出投影
        self.position_gamma = nn.Parameter(torch.zeros(1))
        self.channel_gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch, C, H, W = x.shape
        N = H * W  # 空间位置数
        
        # ========== 位置自注意力 ==========
        # 生成 Q, K, V
        pos_q = self.position_query(x).view(batch,C, N)
        pos_k = self.position_key(x).view(batch,C, N)
        pos_v = self.position_value(x).view(batch,C, N)
        # pos_q = self.position_query(x).view(batch, -1, N).permute(0, 2, 1)  # (B, N, C/8)
        # pos_k = self.position_key(x).view(batch, -1, N)  # (B, C/8, N)
        # pos_v = self.position_value(x).view(batch, -1, N)  # (B, C, N)
        
        # 计算注意力图
        #pos_attn = torch.bmm(pos_q, pos_k)  # (B, N, N)
        pos_attn = torch.bmm(pos_q.permute(0, 2, 1), pos_k)  # (B, N, N)
        pos_attn = F.softmax(pos_attn, dim=-1)
        
        # 加权求和
        pos_out = torch.bmm(pos_v, pos_attn)  # (B, C, N)
        pos_out = pos_out.view(batch, C, H, W)
        pos_out = self.position_gamma * pos_out + x  # 残差连接
        
        # ========== 通道自注意力 ==========
        # # 生成 Q, K, V
        # chan_q = self.channel_query(pos_out).view(batch, C, -1)  # (B, C, N)
        # chan_k = self.channel_key(pos_out).view(batch, C, -1)  # (B, C, N)
        # chan_v = self.channel_value(pos_out).view(batch, C, -1)  # (B, C, N)
        
        # # 计算注意力图
        # chan_attn = torch.bmm(chan_q, chan_k.permute(0, 2, 1))  # (B, C, C)
        # chan_attn = F.softmax(chan_attn, dim=-1)
        chan_x1=x.view(batch,C,N)
        chan_x2=torch.bmm(chan_x1,chan_x1.permute(0,2,1))
        chan_x3=F.softmax(chan_x2, dim=-1)
        chan_x4=torch.bmm(chan_x3,chan_x1)
        chan_out=chan_x4.view(batch,C,H,W)
        # 加权求和
        # chan_out = torch.bmm(chan_attn, chan_v)  # (B, C, N)
        # chan_out = chan_out.view(batch, C, H, W)
        chan_out = self.channel_gamma * chan_out + x  # 残差连接
        dual_out=pos_out+chan_out
        return dual_out


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


if __name__ == "__main__":
    main()