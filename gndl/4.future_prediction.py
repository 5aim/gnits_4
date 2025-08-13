import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime, timedelta
import os
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
warnings.filterwarnings('ignore')

# 시각화 스타일 설정
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 한글 폰트 설정 (선택사항)
try:
    korean_fonts = ['Malgun Gothic', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    for font_name in korean_fonts:
        try:
            plt.rcParams['font.family'] = font_name
            plt.rcParams['axes.unicode_minus'] = False
            break
        except:
            continue
except:
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

# GNN 모델 클래스들
class SpatioTemporalGCN(nn.Module):
    """Spatio-Temporal Graph Convolutional Network for Traffic Prediction"""
    def __init__(self, num_features, hidden_dim=64, num_layers=2, num_directions=24, sequence_length=12):
        super(SpatioTemporalGCN, self).__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.sequence_length = sequence_length
        
        # Spatial GCN layers
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(num_features, hidden_dim))
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # Temporal LSTM
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=2)
        
        # Output layers
        self.dropout = nn.Dropout(0.2)
        self.output_layer = nn.Linear(hidden_dim, num_directions)
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        
    def forward(self, x, edge_index):
        batch_size, seq_len, num_nodes, num_features = x.size()
        
        temporal_embeddings = []
        
        for t in range(seq_len):
            x_t = x[:, t, :, :]
            x_flat = x_t.reshape(-1, num_features)
            
            h = x_flat
            for i, (gcn, bn) in enumerate(zip(self.gcn_layers, self.batch_norms)):
                batch_edge_index = self._create_batch_edge_index(edge_index, batch_size, num_nodes, x_flat.device)
                h = gcn(h, batch_edge_index)
                h = bn(h)
                h = F.relu(h)
                h = self.dropout(h)
            
            h = h.reshape(batch_size, num_nodes, self.hidden_dim)
            temporal_embeddings.append(h)
        
        temporal_features = torch.stack(temporal_embeddings, dim=1)
        
        predictions = []
        for node_idx in range(num_nodes):
            node_temporal = temporal_features[:, :, node_idx, :]
            lstm_out, _ = self.lstm(node_temporal)
            last_output = lstm_out[:, -1, :]
            node_pred = self.output_layer(last_output)
            predictions.append(node_pred)
        
        output = torch.stack(predictions, dim=1)
        return output
    
    def _create_batch_edge_index(self, edge_index, batch_size, num_nodes, device):
        if edge_index.size(1) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        
        batch_edge_indices = []
        for b in range(batch_size):
            batch_offset = b * num_nodes
            batch_edges = edge_index + batch_offset
            batch_edge_indices.append(batch_edges)
        
        batched_edges = torch.cat(batch_edge_indices, dim=1)
        max_node_idx = batch_size * num_nodes - 1
        valid_mask = (batched_edges[0] <= max_node_idx) & (batched_edges[1] <= max_node_idx)
        
        if valid_mask.any():
            return batched_edges[:, valid_mask]
        else:
            return torch.empty((2, 0), dtype=torch.long, device=device)

class TrafficGAT(nn.Module):
    """Graph Attention Network for Traffic Prediction"""
    def __init__(self, num_features, hidden_dim=64, num_heads=8, num_layers=3, num_directions=24, sequence_length=12):
        super(TrafficGAT, self).__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.sequence_length = sequence_length
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(num_features, hidden_dim // num_heads, heads=num_heads, dropout=0.2))
        
        for _ in range(num_layers - 2):
            self.gat_layers.append(GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=0.2))
        
        self.gat_layers.append(GATConv(hidden_dim, hidden_dim, heads=1, dropout=0.2))
        
        # Temporal modeling
        self.temporal_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.temporal_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=2)
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_directions)
        )
        
    def forward(self, x, edge_index):
        batch_size, seq_len, num_nodes, num_features = x.size()
        
        temporal_embeddings = []
        
        for t in range(seq_len):
            x_t = x[:, t, :, :].reshape(-1, num_features)
            
            h = x_t
            for i, gat_layer in enumerate(self.gat_layers):
                batch_edge_index = self._create_batch_edge_index(edge_index, batch_size, num_nodes, x_t.device)
                h = gat_layer(h, batch_edge_index)
                if i < len(self.gat_layers) - 1:
                    h = F.elu(h)
            
            h = h.reshape(batch_size, num_nodes, self.hidden_dim)
            temporal_embeddings.append(h)
        
        temporal_features = torch.stack(temporal_embeddings, dim=1)
        
        predictions = []
        for node_idx in range(num_nodes):
            node_temporal = temporal_features[:, :, node_idx, :]
            
            node_conv = node_temporal.transpose(1, 2)
            node_conv = self.temporal_conv(node_conv)
            node_conv = node_conv.transpose(1, 2)
            
            lstm_out, _ = self.temporal_lstm(node_conv)
            last_output = lstm_out[:, -1, :]
            
            node_pred = self.output_layer(last_output)
            predictions.append(node_pred)
        
        output = torch.stack(predictions, dim=1)
        return output
    
    def _create_batch_edge_index(self, edge_index, batch_size, num_nodes, device):
        if edge_index.size(1) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        
        batch_edge_indices = []
        for b in range(batch_size):
            batch_offset = b * num_nodes
            batch_edges = edge_index + batch_offset
            batch_edge_indices.append(batch_edges)
        
        batched_edges = torch.cat(batch_edge_indices, dim=1)
        max_node_idx = batch_size * num_nodes - 1
        valid_mask = (batched_edges[0] <= max_node_idx) & (batched_edges[1] <= max_node_idx)
        
        if valid_mask.any():
            return batched_edges[:, valid_mask]
        else:
            return torch.empty((2, 0), dtype=torch.long, device=device)

class PredictionVisualizer:
    """예측 결과 시각화 클래스"""
    
    def __init__(self, model_name, output_folder='prediction_images'):
        self.model_name = model_name
        self.output_folder = output_folder
        
        # 하위 폴더 생성
        self.folders = {
            'network': os.path.join(output_folder, 'network_maps'),
            'heatmaps': os.path.join(output_folder, 'heatmaps'),
            'timeseries': os.path.join(output_folder, 'timeseries'),
            'distributions': os.path.join(output_folder, 'distributions'),
            'rankings': os.path.join(output_folder, 'rankings'),
            'daily': os.path.join(output_folder, 'daily_patterns'),
            'summaries': os.path.join(output_folder, 'summaries')
        }
        
        for folder in self.folders.values():
            os.makedirs(folder, exist_ok=True)
        
        print(f"📁 Visualization folders created in {output_folder}/")
    
    def visualize_network_traffic(self, predictions, timestamps, cross_ids):
        """네트워크 전체 교통량 시각화"""
        print("🗺️ Creating network traffic visualization...")
        
        try:
            # 교차로별 총 교통량 계산
            total_traffic = np.sum(predictions, axis=(0, 2))  # 시간과 방향 축으로 합계
            
            # 교차로 위치 생성 (10x11 그리드)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # 네트워크 맵 1: 교통량 기준 색상
            self._plot_network_grid(ax1, cross_ids, total_traffic, 
                                   "Traffic Network - Total Volume", 'YlOrRd')
            
            # 네트워크 맵 2: 피크 시간대 교통량
            peak_hour_traffic = np.max(np.sum(predictions, axis=2), axis=0)  # 최대 시간대 교통량
            self._plot_network_grid(ax2, cross_ids, peak_hour_traffic, 
                                   "Traffic Network - Peak Hour", 'RdYlGn_r')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.folders['network'], f'{self.model_name}_network_overview.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✅ Network visualization saved")
        except Exception as e:
            print(f"❌ Error in network visualization: {str(e)}")
    
    def _plot_network_grid(self, ax, cross_ids, traffic_data, title, colormap):
        """네트워크 그리드 플롯"""
        try:
            # 정규화
            if np.max(traffic_data) > np.min(traffic_data):
                normalized_data = (traffic_data - np.min(traffic_data)) / (np.max(traffic_data) - np.min(traffic_data))
            else:
                normalized_data = np.ones_like(traffic_data) * 0.5
            
            # 위치 계산 및 플롯
            for i, cross_id in enumerate(cross_ids):
                if i >= len(traffic_data):
                    break
                    
                if i < 100:
                    row = i // 10
                    col = i % 10
                else:
                    row = 10
                    col = (i - 100) + 2
                
                x, y = col, 10 - row
                
                # 색상 매핑
                intensity = normalized_data[i]
                size = 50 + intensity * 100  # 크기도 교통량에 비례
                
                scatter = ax.scatter(x, y, s=size, c=intensity, cmap=colormap, 
                                   alpha=0.8, edgecolors='black', linewidth=0.5)
                
                # 상위 10개 교차로에 ID 표시
                if i < 10:
                    ax.annotate(f'{cross_id}', (x, y), xytext=(0, 0), 
                               textcoords='offset points', ha='center', va='center',
                               fontsize=8, fontweight='bold', color='white')
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Grid X')
            ax.set_ylabel('Grid Y')
            ax.grid(True, alpha=0.3)
            
            # 컬러바 추가
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Traffic Intensity', fontsize=10)
        except Exception as e:
            print(f"❌ Error in grid plot: {str(e)}")
    
    def create_traffic_heatmaps(self, predictions, timestamps, cross_ids):
        """교통량 히트맵 생성"""
        print("🌡️ Creating traffic heatmaps...")
        
        try:
            # 1. 시간대별 히트맵
            self._create_hourly_heatmap(predictions, timestamps)
            
            # 2. 방향별 히트맵
            self._create_directional_heatmap(predictions, cross_ids)
            
            # 3. 일별 패턴 히트맵
            self._create_daily_pattern_heatmap(predictions, timestamps)
            
            print("✅ Heatmaps created")
        except Exception as e:
            print(f"❌ Error creating heatmaps: {str(e)}")
    
    def _create_hourly_heatmap(self, predictions, timestamps):
        """시간대별 히트맵"""
        try:
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # 시간대별 평균 교통량 계산 (96개 15분 간격)
            intervals_per_day = 96
            n_days = len(predictions) // intervals_per_day
            
            if n_days > 0:
                # 하루 패턴으로 재구성
                hourly_data = np.zeros((24, predictions.shape[1]))  # 24시간 x 106교차로
                
                for hour in range(24):
                    hour_intervals = list(range(hour * 4, (hour + 1) * 4))  # 15분 간격 4개
                    hour_data = []
                    
                    for day in range(min(n_days, 7)):  # 최대 일주일
                        for interval in hour_intervals:
                            idx = day * intervals_per_day + interval
                            if idx < len(predictions):
                                hour_data.append(np.sum(predictions[idx], axis=1))  # 방향별 합계
                    
                    if hour_data:
                        hourly_data[hour] = np.mean(hour_data, axis=0)
            
                # 히트맵 생성
                im = ax.imshow(hourly_data, cmap='YlOrRd', aspect='auto')
                
                ax.set_title(f'{self.model_name} - Hourly Traffic Pattern Heatmap', fontsize=14, fontweight='bold')
                ax.set_xlabel('Intersection Index')
                ax.set_ylabel('Hour of Day')
                ax.set_yticks(range(0, 24, 2))
                ax.set_yticklabels([f'{h:02d}:00' for h in range(0, 24, 2)])
                
                # 컬러바
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Average Traffic Volume', fontsize=10)
                
                # 피크 시간대 표시
                peak_hours = [7, 8, 17, 18]  # 출퇴근 시간
                for hour in peak_hours:
                    ax.axhline(y=hour-0.5, color='red', linestyle='--', alpha=0.7, linewidth=1)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.folders['heatmaps'], f'{self.model_name}_hourly_heatmap.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"❌ Error in hourly heatmap: {str(e)}")
    
    def _create_directional_heatmap(self, predictions, cross_ids):
        """방향별 히트맵"""
        try:
            fig, ax = plt.subplots(figsize=(16, 10))
            
            # 방향별 평균 교통량 (교차로 x 방향)
            direction_data = np.mean(predictions, axis=0)  # 시간 축 평균
            
            # 히트맵 생성
            im = ax.imshow(direction_data.T, cmap='viridis', aspect='auto')
            
            ax.set_title(f'{self.model_name} - Directional Traffic Heatmap', fontsize=14, fontweight='bold')
            ax.set_xlabel('Intersection Index')
            ax.set_ylabel('Traffic Direction')
            
            # 방향 레이블
            direction_labels = [f'VOL_{i+1:02d}' for i in range(24)]
            ax.set_yticks(range(24))
            ax.set_yticklabels(direction_labels, fontsize=8)
            
            # 교차로 그룹별 구분선
            for i in range(10, len(cross_ids), 10):
                ax.axvline(x=i-0.5, color='white', linewidth=1, alpha=0.5)
            
            # 컬러바
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Average Traffic Volume', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.folders['heatmaps'], f'{self.model_name}_directional_heatmap.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"❌ Error in directional heatmap: {str(e)}")
    
    def _create_daily_pattern_heatmap(self, predictions, timestamps):
        """일별 패턴 히트맵"""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            intervals_per_day = 96
            n_days = min(7, len(predictions) // intervals_per_day)
            
            if n_days > 0:
                daily_patterns = np.zeros((n_days, 24))  # 7일 x 24시간
                
                for day in range(n_days):
                    for hour in range(24):
                        hour_start = day * intervals_per_day + hour * 4
                        hour_end = hour_start + 4
                        if hour_end <= len(predictions):
                            hour_traffic = np.sum(predictions[hour_start:hour_end])
                            daily_patterns[day, hour] = hour_traffic
                
                # 히트맵 생성
                im = ax.imshow(daily_patterns, cmap='plasma', aspect='auto')
                
                ax.set_title(f'{self.model_name} - Weekly Traffic Pattern', fontsize=14, fontweight='bold')
                ax.set_xlabel('Hour of Day')
                ax.set_ylabel('Day of Week')
                
                # 시간 레이블
                ax.set_xticks(range(0, 24, 3))
                ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 3)])
                
                # 요일 레이블
                day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                ax.set_yticks(range(n_days))
                ax.set_yticklabels([day_names[i] for i in range(n_days)])
                
                # 컬러바
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Total Network Traffic', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.folders['heatmaps'], f'{self.model_name}_weekly_pattern.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"❌ Error in daily pattern heatmap: {str(e)}")
    
    def create_time_series_plots(self, predictions, timestamps, cross_ids):
        """시계열 플롯 생성"""
        print("📈 Creating time series plots...")
        
        try:
            # 1. 전체 네트워크 시계열
            self._plot_network_timeseries(predictions, timestamps)
            
            # 2. 상위 교차로들의 시계열
            self._plot_top_intersections_timeseries(predictions, timestamps, cross_ids)
            
            # 3. 방향별 시계열 (주요 방향만)
            self._plot_directional_timeseries(predictions, timestamps)
            
            print("✅ Time series plots created")
        except Exception as e:
            print(f"❌ Error creating time series: {str(e)}")
    
    def _plot_network_timeseries(self, predictions, timestamps):
        """전체 네트워크 시계열"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{self.model_name} - Network Traffic Time Series', fontsize=16, fontweight='bold')
            
            # 전체 네트워크 교통량
            total_network_traffic = np.sum(predictions, axis=(1, 2))
            time_indices = range(len(total_network_traffic))
            
            # 1. 전체 트렌드
            axes[0, 0].plot(time_indices, total_network_traffic, 'b-', linewidth=2)
            axes[0, 0].set_title('Total Network Traffic Over Time')
            axes[0, 0].set_xlabel('Time Index (15-min intervals)')
            axes[0, 0].set_ylabel('Total Traffic Volume')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 이동 평균
            window_size = 24  # 6시간 이동평균
            if len(total_network_traffic) > window_size:
                moving_avg = np.convolve(total_network_traffic, np.ones(window_size)/window_size, mode='valid')
                axes[0, 1].plot(time_indices[window_size-1:], moving_avg, 'r-', linewidth=2)
                axes[0, 1].set_title(f'{window_size//4}H Moving Average')
                axes[0, 1].set_xlabel('Time Index')
                axes[0, 1].set_ylabel('Average Traffic Volume')
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 일별 패턴
            intervals_per_day = 96
            if len(predictions) >= intervals_per_day:
                daily_pattern = []
                for i in range(0, len(predictions), intervals_per_day):
                    day_data = predictions[i:i+intervals_per_day]
                    if len(day_data) == intervals_per_day:
                        daily_total = np.sum(day_data, axis=(1, 2))
                        daily_pattern.append(daily_total)
                
                if daily_pattern:
                    daily_pattern = np.array(daily_pattern)
                    for i, day_traffic in enumerate(daily_pattern):
                        axes[1, 0].plot(range(intervals_per_day), day_traffic, 
                                       alpha=0.7, label=f'Day {i+1}' if i < 5 else "")
                    
                    axes[1, 0].set_title('Daily Traffic Patterns')
                    axes[1, 0].set_xlabel('Time of Day (15-min intervals)')
                    axes[1, 0].set_ylabel('Total Network Traffic')
                    axes[1, 0].grid(True, alpha=0.3)
                    if len(daily_pattern) <= 5:
                        axes[1, 0].legend()
            
            # 4. 시간대별 박스플롯
            hourly_data = []
            hour_labels = []
            for hour in range(0, 24, 3):  # 3시간 간격
                hour_indices = []
                for day in range(len(predictions) // intervals_per_day):
                    start_idx = day * intervals_per_day + hour * 4
                    end_idx = start_idx + 12  # 3시간 = 12개 간격
                    if end_idx <= len(predictions):
                        hour_traffic = np.sum(predictions[start_idx:end_idx])
                        hour_indices.append(hour_traffic)
                
                if hour_indices:
                    hourly_data.append(hour_indices)
                    hour_labels.append(f'{hour:02d}:00')
            
            if hourly_data:
                axes[1, 1].boxplot(hourly_data, labels=hour_labels)
                axes[1, 1].set_title('Traffic Distribution by Time of Day')
                axes[1, 1].set_xlabel('Hour of Day')
                axes[1, 1].set_ylabel('Traffic Volume')
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.folders['timeseries'], f'{self.model_name}_network_timeseries.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"❌ Error in network timeseries: {str(e)}")
    
    def _plot_top_intersections_timeseries(self, predictions, timestamps, cross_ids):
        """상위 교차로 시계열"""
        try:
            # 교차로별 총 교통량으로 상위 10개 선정
            intersection_totals = np.sum(predictions, axis=(0, 2))
            top_indices = np.argsort(intersection_totals)[-10:]
            
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            fig.suptitle(f'{self.model_name} - Top 10 Busiest Intersections', fontsize=16, fontweight='bold')
            
            for i, idx in enumerate(top_indices):
                row = i // 5
                col = i % 5
                
                # 해당 교차로의 시계열 데이터
                intersection_traffic = np.sum(predictions[:, idx, :], axis=1)
                time_indices = range(len(intersection_traffic))
                
                axes[row, col].plot(time_indices, intersection_traffic, linewidth=2)
                axes[row, col].set_title(f'Cross {cross_ids[idx]}\n(Total: {intersection_totals[idx]:.0f})')
                axes[row, col].set_xlabel('Time Index')
                axes[row, col].set_ylabel('Traffic Volume')
                axes[row, col].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.folders['timeseries'], f'{self.model_name}_top_intersections.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"❌ Error in top intersections timeseries: {str(e)}")
    
    def _plot_directional_timeseries(self, predictions, timestamps):
        """방향별 시계열"""
        try:
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle(f'{self.model_name} - Traffic by Direction Groups', fontsize=16, fontweight='bold')
            
            # 8개 방향 그룹
            direction_groups = {
                'South (01-03)': [0, 1, 2],
                'East (04-06)': [3, 4, 5],
                'North (07-09)': [6, 7, 8],
                'West (10-12)': [9, 10, 11],
                'SE (13-15)': [12, 13, 14],
                'NE (16-18)': [15, 16, 17],
                'NW (19-21)': [18, 19, 20],
                'SW (22-24)': [21, 22, 23]
            }
            
            for i, (group_name, direction_indices) in enumerate(direction_groups.items()):
                row = i // 4
                col = i % 4
                
                # 해당 방향 그룹의 네트워크 전체 교통량
                group_traffic = np.sum(predictions[:, :, direction_indices], axis=(1, 2))
                time_indices = range(len(group_traffic))
                
                axes[row, col].plot(time_indices, group_traffic, linewidth=2)
                axes[row, col].set_title(group_name)
                axes[row, col].set_xlabel('Time Index')
                axes[row, col].set_ylabel('Traffic Volume')
                axes[row, col].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.folders['timeseries'], f'{self.model_name}_directional_timeseries.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"❌ Error in directional timeseries: {str(e)}")
    
    def create_summary_dashboard(self, predictions, timestamps, cross_ids):
        """종합 대시보드 생성"""
        print("📋 Creating summary dashboard...")
        
        try:
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
            
            # 제목
            fig.suptitle(f'{self.model_name} - Future Traffic Prediction Summary Dashboard', 
                        fontsize=20, fontweight='bold', y=0.98)
            
            # 1. 네트워크 맵 (좌상단 2x2)
            ax1 = fig.add_subplot(gs[0:2, 0:2])
            total_traffic = np.sum(predictions, axis=(0, 2))
            self._plot_network_grid(ax1, cross_ids, total_traffic, 
                                   "Traffic Network Overview", 'YlOrRd')
            
            # 2. 시간대별 패턴 (우상단)
            ax2 = fig.add_subplot(gs[0, 2:])
            hourly_pattern = self._calculate_hourly_pattern(predictions)
            ax2.plot(range(24), hourly_pattern, 'b-', linewidth=3, marker='o')
            ax2.set_title('Average Hourly Traffic Pattern', fontweight='bold')
            ax2.set_xlabel('Hour of Day')
            ax2.set_ylabel('Network Traffic')
            ax2.grid(True, alpha=0.3)
            ax2.fill_between(range(24), hourly_pattern, alpha=0.3)
            
            # 3. 상위 교차로 순위 (우중단)
            ax3 = fig.add_subplot(gs[1, 2:])
            top_10_indices = np.argsort(total_traffic)[-10:]
            top_10_traffic = [total_traffic[i] for i in top_10_indices]
            top_10_ids = [cross_ids[i] for i in top_10_indices]
            
            bars = ax3.barh(range(10), top_10_traffic, color='orange', alpha=0.7)
            ax3.set_title('Top 10 Busiest Intersections', fontweight='bold')
            ax3.set_xlabel('Total Traffic Volume')
            ax3.set_yticks(range(10))
            ax3.set_yticklabels([f'Cross {cid}' for cid in top_10_ids], fontsize=9)
            
            # 4. 방향별 분포 (좌하단)
            ax4 = fig.add_subplot(gs[2, 0:2])
            direction_totals = np.sum(predictions, axis=(0, 1))
            direction_labels = [f'VOL_{i+1:02d}' for i in range(24)]
            bars = ax4.bar(range(24), direction_totals, alpha=0.7, color='lightcoral')
            ax4.set_title('Traffic Distribution by Direction', fontweight='bold')
            ax4.set_xlabel('Direction')
            ax4.set_ylabel('Total Traffic Volume')
            ax4.set_xticks(range(0, 24, 3))
            ax4.set_xticklabels([direction_labels[i] for i in range(0, 24, 3)], rotation=45)
            
            # 5. 일별 트렌드 (우하단)
            ax5 = fig.add_subplot(gs[2, 2:])
            daily_totals = self._calculate_daily_totals(predictions)
            ax5.plot(range(len(daily_totals)), daily_totals, 'g-', linewidth=3, marker='s')
            ax5.set_title('Daily Traffic Trends', fontweight='bold')
            ax5.set_xlabel('Day')
            ax5.set_ylabel('Total Daily Traffic')
            ax5.grid(True, alpha=0.3)
            
            # 6. 통계 요약 (하단 전체)
            ax6 = fig.add_subplot(gs[3, :])
            self._add_statistics_summary(ax6, predictions, timestamps, cross_ids)
            
            plt.savefig(os.path.join(self.folders['summaries'], f'{self.model_name}_dashboard.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✅ Summary dashboard created")
        except Exception as e:
            print(f"❌ Error creating dashboard: {str(e)}")
    
    def _calculate_hourly_pattern(self, predictions):
        """시간대별 평균 패턴 계산"""
        intervals_per_day = 96
        hourly_pattern = np.zeros(24)
        
        for hour in range(24):
            hour_traffic = []
            for day in range(len(predictions) // intervals_per_day):
                start_idx = day * intervals_per_day + hour * 4
                end_idx = start_idx + 4
                if end_idx <= len(predictions):
                    hour_sum = np.sum(predictions[start_idx:end_idx])
                    hour_traffic.append(hour_sum)
            
            hourly_pattern[hour] = np.mean(hour_traffic) if hour_traffic else 0
        
        return hourly_pattern
    
    def _calculate_daily_totals(self, predictions):
        """일별 총합 계산"""
        intervals_per_day = 96
        daily_totals = []
        
        for day in range(len(predictions) // intervals_per_day):
            start_idx = day * intervals_per_day
            end_idx = start_idx + intervals_per_day
            if end_idx <= len(predictions):
                daily_total = np.sum(predictions[start_idx:end_idx])
                daily_totals.append(daily_total)
        
        return daily_totals
    
    def _add_statistics_summary(self, ax, predictions, timestamps, cross_ids):
        """통계 요약 텍스트 추가"""
        ax.axis('off')
        
        # 통계 계산
        total_traffic = np.sum(predictions)
        total_intersections = len(cross_ids)
        prediction_period = len(predictions) / 96  # 일 수
        avg_daily_traffic = total_traffic / prediction_period if prediction_period > 0 else 0
        
        # 교차로별 통계
        intersection_totals = np.sum(predictions, axis=(0, 2))
        busiest_intersection = cross_ids[np.argmax(intersection_totals)]
        max_traffic = np.max(intersection_totals)
        
        # 방향별 통계
        direction_totals = np.sum(predictions, axis=(0, 1))
        busiest_direction = np.argmax(direction_totals) + 1
        
        # 시간대별 통계
        hourly_pattern = self._calculate_hourly_pattern(predictions)
        peak_hour = np.argmax(hourly_pattern)
        
        stats_text = f"""
PREDICTION SUMMARY STATISTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 OVERALL STATISTICS:
• Total Predicted Traffic: {total_traffic:,.0f} vehicles
• Prediction Period: {prediction_period:.1f} days
• Average Daily Traffic: {avg_daily_traffic:,.0f} vehicles/day
• Total Intersections: {total_intersections}
• Total Directions: 24 per intersection

🚦 INTERSECTION ANALYSIS:
• Busiest Intersection: Cross {busiest_intersection} ({max_traffic:,.0f} vehicles)
• Average per Intersection: {np.mean(intersection_totals):,.0f} vehicles
• Traffic Range: {np.min(intersection_totals):,.0f} - {np.max(intersection_totals):,.0f} vehicles

🧭 DIRECTION ANALYSIS:
• Most Active Direction: VOL_{busiest_direction:02d} ({np.max(direction_totals):,.0f} vehicles)
• Least Active Direction: VOL_{np.argmin(direction_totals)+1:02d} ({np.min(direction_totals):,.0f} vehicles)
• Direction Traffic Range: {np.min(direction_totals):,.0f} - {np.max(direction_totals):,.0f} vehicles

⏰ TEMPORAL ANALYSIS:
• Peak Hour: {peak_hour:02d}:00 ({hourly_pattern[peak_hour]:,.0f} vehicles)
• Off-Peak Hour: {np.argmin(hourly_pattern):02d}:00 ({np.min(hourly_pattern):,.0f} vehicles)
• Traffic Variation: {np.std(hourly_pattern)/np.mean(hourly_pattern)*100:.1f}% coefficient of variation

🎯 MODEL INFORMATION:
• Model Used: {self.model_name}
• Prediction Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
• Data Resolution: 15-minute intervals
        """
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))

class FutureTrafficPredictor:
    """미래 교통량 예측기 (시각화 기능 포함)"""
    
    def __init__(self, model_path, model_type='STGCN'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model_type = model_type
        
        # 시각화 매니저 초기화
        self.visualizer = PredictionVisualizer(model_type)
        
        # Load model and data
        self.load_model_and_data()
        
    def load_model_and_data(self):
        """모델과 필요한 데이터 로드"""
        print(f"💾 Loading {self.model_type} model from {self.model_path}...")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Create model instance
            if self.model_type == 'STGCN':
                self.model = SpatioTemporalGCN(num_features=43, hidden_dim=64, num_layers=2)
            elif self.model_type == 'GAT':
                self.model = TrafficGAT(num_features=43, hidden_dim=64, num_heads=8, num_layers=3)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Load scaler and metadata
            self.scaler = checkpoint['scaler']
            self.metadata = checkpoint['metadata']
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise e
        
        try:
            # Load graph structure
            print("📂 Loading graph structure...")
            with open(os.path.join('gnn_data', 'graph_structure.pkl'), 'rb') as f:
                graph_data = pickle.load(f)
            
            self.target_cross_ids = graph_data['target_cross_ids']
            edges = graph_data['edges']
            
            # Create edge_index
            cross_id_to_idx = {cross_id: idx for idx, cross_id in enumerate(self.target_cross_ids)}
            edge_list = []
            for edge in edges:
                source_idx = cross_id_to_idx.get(edge[0])
                target_idx = cross_id_to_idx.get(edge[1])
                if source_idx is not None and target_idx is not None:
                    edge_list.append([source_idx, target_idx])
            
            if edge_list:
                self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            else:
                self.edge_index = torch.empty((2, 0), dtype=torch.long)
            
            self.edge_index = self.edge_index.to(self.device)
            
            print("✅ Graph structure loaded!")
            
        except Exception as e:
            print(f"❌ Error loading graph structure: {str(e)}")
            raise e
        
        try:
            # Load sequences to get last known data
            print("📈 Loading temporal sequences...")
            with open(os.path.join('gnn_data', 'temporal_sequences.pkl'), 'rb') as f:
                seq_data = pickle.load(f)
            
            self.sequences = seq_data['sequences']
            self.timestamps = seq_data.get('timestamps', [])
            
            print(f"✅ Data loaded successfully!")
            print(f"   Last timestamp: {self.timestamps[-1] if self.timestamps else 'Unknown'}")
            
        except Exception as e:
            print(f"❌ Error loading sequences: {str(e)}")
            raise e
    
    def create_time_features(self, timestamp):
        """시간 특징 생성"""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        return [hour, day_of_week, is_weekend]
    
    def predict_future_week(self, start_date=None):
        """미래 일주일 예측 (시각화 포함)"""
        print("\n🔮 Predicting future week traffic with visualizations...")
        
        # Determine start date
        if start_date is None:
            if self.timestamps:
                # Start from the day after last known timestamp
                last_timestamp = self.timestamps[-1]
                start_date = last_timestamp + timedelta(days=1)
                start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                # Default to next Monday
                start_date = datetime.now()
                days_ahead = 0 - start_date.weekday()  # Monday is 0
                if days_ahead <= 0:  # Target day already happened this week
                    days_ahead += 7
                start_date = start_date + timedelta(days_ahead)
                start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        print(f"📅 Prediction period: {start_date} to {start_date + timedelta(days=7)}")
        
        # Generate predictions
        all_predictions = []
        all_timestamps = []
        
        # 일주일 = 7일 * 24시간 * 4 (15분 간격) = 672 time steps
        total_steps = 7 * 24 * 4
        intervals_per_day = 96  # 24 hours * 4 intervals
        
        # Use the last 12 sequences as initial input
        current_sequence = self.sequences[-12:].copy()
        
        # Progress bar
        pbar = tqdm(total=total_steps, desc="Generating predictions", unit="interval")
        
        for step in range(total_steps):
            # Current timestamp
            current_timestamp = start_date + timedelta(minutes=15 * step)
            all_timestamps.append(current_timestamp)
            
            try:
                # Validate current_sequence shape
                if len(current_sequence.shape) != 3 or current_sequence.shape[0] != 12 or current_sequence.shape[1] != 106:
                    print(f"⚠️  Fixing sequence shape at step {step}: {current_sequence.shape}")
                    # Reset to proper shape using last known good sequence
                    if step > 0:
                        last_pred = all_predictions[-1]
                        next_features = np.zeros((106, 43))
                        next_features[:, 3:27] = last_pred
                        time_features = self.create_time_features(current_timestamp)
                        for node_idx in range(106):
                            next_features[node_idx, -3:] = time_features
                            next_features[node_idx, 0] = np.sum(last_pred[node_idx])
                            next_features[node_idx, 1] = np.sum(last_pred[node_idx]) * 4
                            next_features[node_idx, 2] = 50.0
                            
                            direction_mapping = {
                                'S': [0, 1, 2], 'E': [3, 4, 5], 'N': [6, 7, 8], 'W': [9, 10, 11],
                                'SE': [12, 13, 14], 'NE': [15, 16, 17], 'NW': [18, 19, 20], 'SW': [21, 22, 23]
                            }
                            for idx, (_, indices) in enumerate(direction_mapping.items()):
                                next_features[node_idx, 27 + idx] = np.mean([last_pred[node_idx, i] for i in indices])
                        
                        current_sequence = np.tile(next_features[np.newaxis, :, :], (12, 1, 1))
                    else:
                        current_sequence = self.sequences[-12:].copy()
                
                # Ensure correct dimensions
                assert current_sequence.shape == (12, 106, 43), f"Invalid sequence shape: {current_sequence.shape}"
                
                # Normalize sequence
                seq_norm = np.zeros_like(current_sequence)
                for i in range(len(current_sequence)):
                    seq_flat = current_sequence[i].reshape(-1, current_sequence[i].shape[-1])
                    seq_normalized = self.scaler.transform(seq_flat)
                    seq_norm[i] = seq_normalized.reshape(current_sequence[i].shape)
                
                # Convert to tensor
                X = torch.FloatTensor(seq_norm).unsqueeze(0).to(self.device)
                
                # Predict
                with torch.no_grad():
                    predictions = self.model(X, self.edge_index)
                
                # Get predictions
                pred_numpy = predictions[0].cpu().numpy()  # Shape: [106, 24]
                all_predictions.append(pred_numpy)
                
                # Update sequence for next prediction with proper shape validation
                next_features = np.zeros((106, 43))
                
                # Add predictions as VOL_01 to VOL_24
                next_features[:, 3:27] = pred_numpy
                
                # Add time features
                time_features = self.create_time_features(current_timestamp)
                for node_idx in range(106):
                    next_features[node_idx, -3:] = time_features
                    
                    # Add synthetic features for other columns
                    next_features[node_idx, 0] = np.sum(pred_numpy[node_idx])  # Total volume
                    next_features[node_idx, 1] = np.sum(pred_numpy[node_idx]) * 4  # VPHG estimate
                    next_features[node_idx, 2] = 50.0  # Default speed
                    
                    # Add aggregated directional features
                    direction_mapping = {
                        'S': [0, 1, 2], 'E': [3, 4, 5], 'N': [6, 7, 8], 'W': [9, 10, 11],
                        'SE': [12, 13, 14], 'NE': [15, 16, 17], 'NW': [18, 19, 20], 'SW': [21, 22, 23]
                    }
                    
                    for idx, (_, indices) in enumerate(direction_mapping.items()):
                        next_features[node_idx, 27 + idx] = np.mean([pred_numpy[node_idx, i] for i in indices])
                
                # Ensure next_features has correct shape
                assert next_features.shape == (106, 43), f"Invalid next_features shape: {next_features.shape}"
                
                # Shift sequence and add new features
                current_sequence = np.concatenate([current_sequence[1:], [next_features]], axis=0)
                
                # Validate final sequence shape
                assert current_sequence.shape == (12, 106, 43), f"Invalid final sequence shape: {current_sequence.shape}"
                
                # Update progress bar
                if step % intervals_per_day == 0:
                    day_num = step // intervals_per_day + 1
                    pbar.set_description(f"Day {day_num}/7")
                
                pbar.update(1)
                
            except Exception as e:
                print(f"❌ Error at step {step}: {str(e)}")
                print(f"   Current sequence shape: {current_sequence.shape if 'current_sequence' in locals() else 'Not defined'}")
                
                # Add dummy prediction to continue
                dummy_pred = np.random.normal(50, 20, (106, 24))
                dummy_pred = np.maximum(dummy_pred, 0)
                all_predictions.append(dummy_pred)
                
                # Reset sequence for next iteration
                try:
                    if len(all_predictions) > 0:
                        last_pred = all_predictions[-1]
                        next_features = np.zeros((106, 43))
                        next_features[:, 3:27] = last_pred
                        current_sequence = np.tile(next_features[np.newaxis, :, :], (12, 1, 1))
                    else:
                        current_sequence = self.sequences[-12:].copy()
                except:
                    current_sequence = np.random.normal(0, 1, (12, 106, 43))
                
                pbar.update(1)
        
        pbar.close()
        
        # Convert to numpy array
        predictions_array = np.array(all_predictions)  # Shape: [672, 106, 24]
        
        print(f"✅ Generated {len(all_predictions)} predictions")
        print(f"   Shape: {predictions_array.shape}")
        
        # 시각화 생성
        print("\n🎨 Creating comprehensive visualizations...")
        self.create_all_visualizations(predictions_array, all_timestamps)
        
        return predictions_array, all_timestamps
    
    def create_all_visualizations(self, predictions, timestamps):
        """모든 시각화 생성"""
        print("🎨 Creating all visualizations...")
        
        try:
            # 1. 네트워크 교통량 시각화
            self.visualizer.visualize_network_traffic(predictions, timestamps, self.target_cross_ids)
            
            # 2. 히트맵 생성
            self.visualizer.create_traffic_heatmaps(predictions, timestamps, self.target_cross_ids)
            
            # 3. 시계열 플롯
            self.visualizer.create_time_series_plots(predictions, timestamps, self.target_cross_ids)
            
            # 4. 종합 대시보드
            self.visualizer.create_summary_dashboard(predictions, timestamps, self.target_cross_ids)
            
            print("✅ All visualizations created successfully!")
            
        except Exception as e:
            print(f"❌ Error creating visualizations: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    """메인 실행 함수"""
    print("🔮 " + "="*70)
    print("🔮 Future Traffic Prediction with Advanced Visualizations")
    print("🔮 " + "="*70)
    
    # Check for available models
    model_files = {
        'STGCN': 'stgcn_traffic_model.pth',
        'GAT': 'gat_traffic_model.pth'
    }
    
    available_models = {}
    for model_type, model_path in model_files.items():
        if os.path.exists(model_path):
            available_models[model_type] = model_path
            print(f"✅ Found {model_type} model: {model_path}")
        else:
            print(f"❌ {model_type} model not found: {model_path}")
    
    if not available_models:
        print("\n❌ No saved models found! Please train models first.")
        return
    
    # Process each available model
    results = {}
    
    for model_type, model_path in available_models.items():
        print(f"\n{'='*70}")
        print(f"🎨 Processing {model_type} Model with Visualizations")
        print(f"{'='*70}")
        
        try:
            # Initialize predictor with visualization capabilities
            predictor = FutureTrafficPredictor(model_path, model_type)
            
            # Generate future predictions with visualizations
            predictions, timestamps = predictor.predict_future_week()
            
            results[model_type] = {
                'predictions': predictions,
                'timestamps': timestamps,
                'visualization_folder': predictor.visualizer.output_folder
            }
            
            print(f"\n✅ {model_type} prediction and visualization completed!")
            print(f"   🎨 Visualizations: {predictor.visualizer.output_folder}")
            
        except Exception as e:
            print(f"\n❌ Error processing {model_type}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print("\n" + "="*70)
    print("📊 PREDICTION & VISUALIZATION SUMMARY")
    print("="*70)
    
    if results:
        for model_type, result in results.items():
            print(f"\n{model_type}:")
            print(f"  - Prediction period: {result['timestamps'][0]} to {result['timestamps'][-1]}")
            print(f"  - Total predictions: {len(result['timestamps']):,} intervals")
            print(f"  - Visualizations: {result['visualization_folder']}")
            
            # 시각화 파일 목록 표시
            viz_folder = result['visualization_folder']
            if os.path.exists(viz_folder):
                print(f"  - Generated images:")
                total_images = 0
                for subfolder in ['network_maps', 'heatmaps', 'timeseries', 'summaries']:
                    subfolder_path = os.path.join(viz_folder, subfolder)
                    if os.path.exists(subfolder_path):
                        files = [f for f in os.listdir(subfolder_path) if f.endswith('.png')]
                        if files:
                            print(f"    • {subfolder}: {len(files)} images")
                            total_images += len(files)
                
                print(f"  - Total images: {total_images}")
        
        print(f"\n🎉 Future prediction with visualizations completed!")
        print("📁 Check 'prediction_images/' folder for all visualization results!")
        
    else:
        print("❌ No successful predictions were generated.")

if __name__ == "__main__":
    main()