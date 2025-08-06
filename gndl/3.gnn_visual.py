import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')  # 백엔드 설정 - GUI 없이 저장만
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ============= GNN Models (필요한 모델 정의) =============

class SpatioTemporalGCN(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network for Traffic Prediction
    """
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
    """
    Graph Attention Network for Traffic Prediction
    """
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

# ============= Visualization Class =============

class ModelVisualizer:
    """저장된 모델로부터 시각화 생성"""
    
    def __init__(self, output_folder='visualization_results'):
        self.output_folder = output_folder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output folders
        os.makedirs(output_folder, exist_ok=True)
        self.folders = {
            'accuracy': os.path.join(output_folder, 'accuracy_plots'),
            'error': os.path.join(output_folder, 'error_plots'),
            'comparison': os.path.join(output_folder, 'comparison_plots'),
            'time_series': os.path.join(output_folder, 'time_series_plots')
        }
        
        for folder in self.folders.values():
            os.makedirs(folder, exist_ok=True)
        
        print(f"📁 Visualization folders created in {output_folder}/")
    
    def load_model_and_data(self, model_path, model_type='STGCN'):
        """Load saved model and prepare data"""
        print(f"\n💾 Loading model from {model_path}...")
        
        # Load checkpoint with weights_only=False (for sklearn objects)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Create model instance
        if model_type == 'STGCN':
            model = SpatioTemporalGCN(num_features=43, hidden_dim=64, num_layers=2)
        elif model_type == 'GAT':
            model = TrafficGAT(num_features=43, hidden_dim=64, num_heads=8, num_layers=3)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        # Load scaler and metadata
        scaler = checkpoint['scaler']
        metadata = checkpoint['metadata']
        
        print(f"✅ Model loaded successfully!")
        
        # Load test data
        print("📂 Loading test data...")
        
        # Load graph structure
        with open(os.path.join('gnn_data', 'graph_structure.pkl'), 'rb') as f:
            graph_data = pickle.load(f)
        
        target_cross_ids = graph_data['target_cross_ids']
        edges = graph_data['edges']
        
        # Create edge_index
        cross_id_to_idx = {cross_id: idx for idx, cross_id in enumerate(target_cross_ids)}
        edge_list = []
        for edge in edges:
            source_idx = cross_id_to_idx.get(edge[0])
            target_idx = cross_id_to_idx.get(edge[1])
            if source_idx is not None and target_idx is not None:
                edge_list.append([source_idx, target_idx])
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        edge_index = edge_index.to(self.device)
        
        # Load temporal sequences
        with open(os.path.join('gnn_data', 'temporal_sequences.pkl'), 'rb') as f:
            seq_data = pickle.load(f)
        
        sequences = seq_data['sequences']
        targets = seq_data['targets']
        timestamps = seq_data.get('timestamps', None)
        
        print(f"✅ Data loaded: {len(sequences)} sequences")
        
        return model, scaler, edge_index, sequences, targets, timestamps, target_cross_ids
    
    def generate_predictions(self, model, scaler, edge_index, sequences, targets, sample_size=500):
        """Generate predictions using the loaded model"""
        print(f"\n🔮 Generating predictions on {sample_size} samples...")
        
        # Take last samples as test set
        test_sequences = sequences[-sample_size:]
        test_targets = targets[-sample_size:]
        
        # Normalize sequences
        test_sequences_norm = np.zeros_like(test_sequences)
        for i in range(len(test_sequences)):
            seq_flat = test_sequences[i].reshape(-1, test_sequences[i].shape[-1])
            seq_normalized = scaler.transform(seq_flat)
            test_sequences_norm[i] = seq_normalized.reshape(test_sequences[i].shape)
        
        # Convert to tensors
        X_test = torch.FloatTensor(test_sequences_norm).to(self.device)
        y_test = torch.FloatTensor(test_targets).to(self.device)
        
        # Generate predictions
        model.eval()
        with torch.no_grad():
            predictions = model(X_test, edge_index)
        
        # Convert to numpy
        y_true = y_test.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        
        # Calculate metrics
        mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
        mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true.flatten(), y_pred.flatten())
        
        print(f"✅ Predictions generated!")
        print(f"   MAE:  {mae:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   R²:   {r2:.4f}")
        
        return y_true, y_pred, {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}
    
    def create_all_visualizations(self, y_true, y_pred, metrics, model_name, cross_ids):
        """Create comprehensive visualizations"""
        print(f"\n📊 Creating visualizations for {model_name}...")
        
        try:
            # 1. Accuracy visualizations
            print("  1️⃣ Creating accuracy analysis...")
            self.plot_accuracy_analysis(y_true, y_pred, metrics, model_name)
            print("     ✅ Accuracy analysis completed")
            
            # 2. Error visualizations
            print("  2️⃣ Creating error analysis...")
            self.plot_error_analysis(y_true, y_pred, model_name)
            print("     ✅ Error analysis completed")
            
            # 3. Direction-wise performance
            print("  3️⃣ Creating directional performance analysis...")
            self.plot_directional_performance(y_true, y_pred, model_name)
            print("     ✅ Directional performance completed")
            
            # 4. Time series comparison
            print("  4️⃣ Creating time series comparison...")
            self.plot_time_series_comparison(y_true, y_pred, model_name)
            print("     ✅ Time series comparison completed")
            
            # 5. Intersection performance
            print("  5️⃣ Creating intersection performance analysis...")
            self.plot_intersection_performance(y_true, y_pred, model_name, cross_ids)
            print("     ✅ Intersection performance completed")
            
            # 6. Peak hour analysis
            print("  6️⃣ Creating peak hour analysis...")
            self.plot_peak_hour_analysis(y_true, y_pred, model_name)
            print("     ✅ Peak hour analysis completed")
            
            print(f"\n✅ All visualizations created for {model_name}!")
            
        except Exception as e:
            print(f"\n❌ Error creating visualizations: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def plot_accuracy_analysis(self, y_true, y_pred, metrics, model_name):
        """정확도 분석 시각화"""
        plt.figure(figsize=(20, 12))
        
        # 1. Scatter plot
        plt.subplot(2, 3, 1)
        sample_size = min(5000, y_true.size)
        idx = np.random.choice(y_true.size, sample_size, replace=False)
        y_true_sample = y_true.flatten()[idx]
        y_pred_sample = y_pred.flatten()[idx]
        
        plt.scatter(y_true_sample, y_pred_sample, alpha=0.5, s=10)
        plt.plot([0, y_true_sample.max()], [0, y_true_sample.max()], 'r--', linewidth=2)
        plt.xlabel('Actual Traffic Volume')
        plt.ylabel('Predicted Traffic Volume')
        plt.title(f'{model_name} - Predictions vs Actual')
        plt.grid(True, alpha=0.3)
        
        # 2. R² by direction
        plt.subplot(2, 3, 2)
        direction_r2 = []
        for i in range(24):
            dir_r2 = r2_score(y_true[:, :, i].flatten(), y_pred[:, :, i].flatten())
            direction_r2.append(dir_r2)
        
        plt.bar(range(24), direction_r2, color='skyblue', alpha=0.7)
        plt.axhline(y=metrics['r2'], color='red', linestyle='--', label=f'Overall R²: {metrics["r2"]:.3f}')
        plt.xlabel('Direction (VOL_01 to VOL_24)')
        plt.ylabel('R² Score')
        plt.title('R² Score by Direction')
        plt.legend()
        plt.xticks(range(0, 24, 2))
        plt.grid(True, alpha=0.3)
        
        # 3. MAE heatmap
        plt.subplot(2, 3, 3)
        mae_matrix = np.zeros((y_true.shape[1], 24))
        for i in range(y_true.shape[1]):
            for j in range(24):
                mae_matrix[i, j] = mean_absolute_error(
                    y_true[:, i, j], y_pred[:, i, j]
                )
        
        sns.heatmap(mae_matrix[:20, :], cmap='YlOrRd', cbar_kws={'label': 'MAE'})
        plt.xlabel('Direction')
        plt.ylabel('Intersection (Sample)')
        plt.title('MAE Heatmap (Sample Intersections)')
        
        # 4. Cumulative accuracy
        plt.subplot(2, 3, 4)
        abs_errors = np.abs(y_pred - y_true).flatten()
        sorted_errors = np.sort(abs_errors)
        percentiles = np.arange(len(sorted_errors)) / len(sorted_errors) * 100
        
        plt.plot(sorted_errors, percentiles, 'g-', linewidth=2)
        plt.xlabel('Absolute Error')
        plt.ylabel('Percentile (%)')
        plt.title('Cumulative Error Distribution')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, np.percentile(sorted_errors, 95))
        
        # Add markers
        for p in [50, 80, 90, 95]:
            val = np.percentile(sorted_errors, p)
            plt.axvline(val, color='red', linestyle='--', alpha=0.5)
            plt.text(val, p+2, f'{p}%: {val:.1f}', rotation=90, fontsize=8)
        
        # 5. Prediction confidence
        plt.subplot(2, 3, 5)
        relative_errors = np.abs(y_pred - y_true) / (y_true + 1e-8) * 100
        confidence_thresholds = [10, 20, 30, 50, 100]
        confidence_levels = []
        
        for threshold in confidence_thresholds:
            conf_level = (relative_errors <= threshold).sum() / relative_errors.size * 100
            confidence_levels.append(conf_level)
        
        plt.plot(confidence_thresholds, confidence_levels, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Error Threshold (%)')
        plt.ylabel('Prediction Coverage (%)')
        plt.title('Model Confidence Analysis')
        plt.grid(True, alpha=0.3)
        
        for i, (thresh, conf) in enumerate(zip(confidence_thresholds, confidence_levels)):
            plt.annotate(f'{conf:.1f}%', (thresh, conf), textcoords="offset points", 
                        xytext=(0,5), ha='center', fontsize=8)
        
        # 6. Summary
        plt.subplot(2, 3, 6)
        summary_text = f"""
        {model_name} Performance Summary
        
        Overall Metrics:
        - MAE: {metrics['mae']:.4f}
        - RMSE: {metrics['rmse']:.4f}
        - R²: {metrics['r2']:.4f}
        - MAPE: {np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100:.2f}%
        
        Coverage Analysis:
        - Error < 10: {(abs_errors < 10).sum() / abs_errors.size * 100:.1f}%
        - Error < 20: {(abs_errors < 20).sum() / abs_errors.size * 100:.1f}%
        - Error < 50: {(abs_errors < 50).sum() / abs_errors.size * 100:.1f}%
        
        Direction Performance:
        - Best R²: {max(direction_r2):.3f} (VOL_{np.argmax(direction_r2)+1:02d})
        - Worst R²: {min(direction_r2):.3f} (VOL_{np.argmin(direction_r2)+1:02d})
        """
        
        plt.text(0.05, 0.5, summary_text, fontsize=11, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        plt.axis('off')
        
        plt.suptitle(f'{model_name} - Accuracy Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folders['accuracy'], f'{model_name}_accuracy_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_error_analysis(self, y_true, y_pred, model_name):
        """오차 분석 시각화"""
        errors = y_pred - y_true
        abs_errors = np.abs(errors)
        
        plt.figure(figsize=(20, 10))
        
        # 1. Error distribution
        plt.subplot(2, 3, 1)
        plt.hist(errors.flatten(), bins=100, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        plt.text(0.02, 0.98, f'Mean: {np.mean(errors):.2f}\nStd: {np.std(errors):.2f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Error by traffic volume
        plt.subplot(2, 3, 2)
        plt.scatter(y_true.flatten()[:5000], errors.flatten()[:5000], alpha=0.3, s=5)
        plt.axhline(0, color='red', linestyle='--', linewidth=1)
        plt.xlabel('Actual Traffic Volume')
        plt.ylabel('Prediction Error')
        plt.title('Error vs Traffic Volume')
        plt.grid(True, alpha=0.3)
        
        # 3. Absolute error by direction
        plt.subplot(2, 3, 3)
        direction_mae = []
        for i in range(24):
            dir_mae = mean_absolute_error(y_true[:, :, i].flatten(), y_pred[:, :, i].flatten())
            direction_mae.append(dir_mae)
        
        bars = plt.bar(range(24), direction_mae, color='orange', alpha=0.7)
        best_idx = np.argmin(direction_mae)
        worst_idx = np.argmax(direction_mae)
        bars[best_idx].set_color('green')
        bars[worst_idx].set_color('red')
        
        plt.xlabel('Direction')
        plt.ylabel('MAE')
        plt.title('Mean Absolute Error by Direction')
        plt.xticks(range(0, 24, 2))
        plt.grid(True, alpha=0.3)
        
        # 4. Error percentiles
        plt.subplot(2, 3, 4)
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        percentile_values = [np.percentile(abs_errors, p) for p in percentiles]
        
        plt.bar(range(len(percentiles)), percentile_values, color='purple', alpha=0.7)
        plt.xticks(range(len(percentiles)), [f'{p}%' for p in percentiles])
        plt.xlabel('Percentile')
        plt.ylabel('Absolute Error')
        plt.title('Error Percentiles')
        plt.grid(True, alpha=0.3)
        
        # 5. Relative error distribution
        plt.subplot(2, 3, 5)
        relative_errors = np.abs(errors) / (y_true + 1e-8) * 100
        plt.hist(relative_errors.flatten()[relative_errors.flatten() < 200], 
                bins=50, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel('Relative Error (%)')
        plt.ylabel('Frequency')
        plt.title('Relative Error Distribution')
        plt.xlim(0, 200)
        plt.grid(True, alpha=0.3)
        
        # 6. Error statistics
        plt.subplot(2, 3, 6)
        error_stats_text = f"""
        Error Statistics
        
        Absolute Error:
        - Mean: {np.mean(abs_errors):.4f}
        - Median: {np.median(abs_errors):.4f}
        - Std: {np.std(abs_errors):.4f}
        - Max: {np.max(abs_errors):.4f}
        
        Relative Error:
        - Mean: {np.mean(relative_errors):.2f}%
        - Median: {np.median(relative_errors):.2f}%
        
        Error Ranges:
        - [-10, 10]: {((errors >= -10) & (errors <= 10)).sum() / errors.size * 100:.1f}%
        - [-20, 20]: {((errors >= -20) & (errors <= 20)).sum() / errors.size * 100:.1f}%
        - [-50, 50]: {((errors >= -50) & (errors <= 50)).sum() / errors.size * 100:.1f}%
        """
        
        plt.text(0.05, 0.5, error_stats_text, fontsize=11, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        plt.axis('off')
        
        plt.suptitle(f'{model_name} - Error Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folders['error'], f'{model_name}_error_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_directional_performance(self, y_true, y_pred, model_name):
        """방향별 성능 분석"""
        plt.figure(figsize=(20, 10))
        
        # Direction groups
        direction_groups = {
            'S': [0, 1, 2], 'E': [3, 4, 5], 'N': [6, 7, 8], 'W': [9, 10, 11],
            'SE': [12, 13, 14], 'NE': [15, 16, 17], 'NW': [18, 19, 20], 'SW': [21, 22, 23]
        }
        
        # 1. Direction group performance
        plt.subplot(2, 2, 1)
        group_metrics = {}
        for group_name, indices in direction_groups.items():
            group_true = np.concatenate([y_true[:, :, i].flatten() for i in indices])
            group_pred = np.concatenate([y_pred[:, :, i].flatten() for i in indices])
            group_metrics[group_name] = {
                'MAE': mean_absolute_error(group_true, group_pred),
                'R2': r2_score(group_true, group_pred)
            }
        
        groups = list(group_metrics.keys())
        mae_vals = [group_metrics[g]['MAE'] for g in groups]
        
        plt.bar(groups, mae_vals, color='coral', alpha=0.7)
        plt.xlabel('Direction Group')
        plt.ylabel('MAE')
        plt.title('Performance by Direction Group')
        plt.grid(True, alpha=0.3)
        
        # 2. Direction heatmap
        plt.subplot(2, 2, 2)
        direction_performance = np.zeros((6, 4))
        for i in range(24):
            row = i // 4
            col = i % 4
            mae = mean_absolute_error(y_true[:, :, i].flatten(), y_pred[:, :, i].flatten())
            direction_performance[row, col] = mae
        
        sns.heatmap(direction_performance, annot=True, fmt='.2f', cmap='RdYlGn_r',
                   xticklabels=['1', '2', '3', '4'],
                   yticklabels=['VOL_01-04', 'VOL_05-08', 'VOL_09-12', 
                               'VOL_13-16', 'VOL_17-20', 'VOL_21-24'])
        plt.title('MAE Heatmap by Direction')
        
        # 3. Sample time series for each direction group
        plt.subplot(2, 2, 3)
        sample_intersection = 0
        sample_time = min(100, y_true.shape[0])
        
        for group_name, indices in list(direction_groups.items())[:4]:  # First 4 groups
            group_actual = np.mean([y_true[:sample_time, sample_intersection, i] for i in indices], axis=0)
            plt.plot(group_actual, label=f'{group_name}', linewidth=2)
        
        plt.xlabel('Time Steps')
        plt.ylabel('Traffic Volume')
        plt.title('Traffic Pattern by Main Directions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Direction summary
        plt.subplot(2, 2, 4)
        summary_text = "Direction Performance Summary\n\n"
        
        # Best and worst directions
        direction_mae = [(i, mean_absolute_error(y_true[:, :, i].flatten(), 
                                               y_pred[:, :, i].flatten())) 
                        for i in range(24)]
        direction_mae.sort(key=lambda x: x[1])
        
        summary_text += "Best Performing Directions:\n"
        for i, (dir_idx, mae) in enumerate(direction_mae[:3]):
            summary_text += f"  {i+1}. VOL_{dir_idx+1:02d}: MAE = {mae:.2f}\n"
        
        summary_text += "\nWorst Performing Directions:\n"
        for i, (dir_idx, mae) in enumerate(direction_mae[-3:]):
            summary_text += f"  {i+1}. VOL_{dir_idx+1:02d}: MAE = {mae:.2f}\n"
        
        summary_text += "\nDirection Group Performance:\n"
        for group, metrics in group_metrics.items():
            summary_text += f"  {group}: MAE={metrics['MAE']:.2f}, R²={metrics['R2']:.3f}\n"
        
        plt.text(0.05, 0.5, summary_text, fontsize=10, verticalalignment='center',
                fontfamily='monospace')
        plt.axis('off')
        
        plt.suptitle(f'{model_name} - Directional Performance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folders['comparison'], f'{model_name}_directional_performance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_time_series_comparison(self, y_true, y_pred, model_name):
        """시계열 비교 분석"""
        plt.figure(figsize=(20, 12))
        
        # Sample different intersections and directions
        sample_intersections = [0, 10, 20, 30]
        sample_directions = [0, 6, 12, 18]  # S, N, SE, NW
        
        plot_idx = 1
        for int_idx in sample_intersections[:2]:
            for dir_idx in sample_directions[:2]:
                plt.subplot(2, 2, plot_idx)
                
                time_steps = min(200, y_true.shape[0])
                actual = y_true[:time_steps, int_idx, dir_idx]
                predicted = y_pred[:time_steps, int_idx, dir_idx]
                
                plt.plot(actual, 'b-', label='Actual', linewidth=2, alpha=0.8)
                plt.plot(predicted, 'r--', label='Predicted', linewidth=2, alpha=0.8)
                
                # Add error band
                error = np.abs(predicted - actual)
                plt.fill_between(range(time_steps), 
                               predicted - error, predicted + error, 
                               alpha=0.2, color='red')
                
                mae = mean_absolute_error(actual, predicted)
                r2 = r2_score(actual, predicted)
                
                plt.xlabel('Time Steps (15min intervals)')
                plt.ylabel('Traffic Volume')
                plt.title(f'Intersection {int_idx}, Direction VOL_{dir_idx+1:02d}\nMAE: {mae:.2f}, R²: {r2:.3f}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plot_idx += 1
        
        plt.suptitle(f'{model_name} - Time Series Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folders['time_series'], f'{model_name}_time_series.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_intersection_performance(self, y_true, y_pred, model_name, cross_ids):
        """교차로별 성능 분석"""
        plt.figure(figsize=(20, 10))
        
        # Calculate performance metrics for each intersection
        intersection_metrics = []
        for i in range(y_true.shape[1]):
            mae = mean_absolute_error(y_true[:, i, :].flatten(), y_pred[:, i, :].flatten())
            rmse = np.sqrt(mean_squared_error(y_true[:, i, :].flatten(), y_pred[:, i, :].flatten()))
            r2 = r2_score(y_true[:, i, :].flatten(), y_pred[:, i, :].flatten())
            
            intersection_metrics.append({
                'idx': i,
                'cross_id': cross_ids[i] if i < len(cross_ids) else f'Cross_{i}',
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            })
        
        # Sort by MAE
        intersection_metrics.sort(key=lambda x: x['mae'])
        
        # 1. Top 20 best intersections
        plt.subplot(2, 2, 1)
        top_20 = intersection_metrics[:20]
        cross_names = [str(x['cross_id']) for x in top_20]
        mae_values = [x['mae'] for x in top_20]
        
        plt.bar(range(20), mae_values, color='green', alpha=0.7)
        plt.xticks(range(20), cross_names, rotation=45, ha='right')
        plt.xlabel('Intersection ID')
        plt.ylabel('MAE')
        plt.title('Top 20 Best Performing Intersections')
        plt.grid(True, alpha=0.3)
        
        # 2. Bottom 20 worst intersections
        plt.subplot(2, 2, 2)
        bottom_20 = intersection_metrics[-20:]
        cross_names = [str(x['cross_id']) for x in bottom_20]
        mae_values = [x['mae'] for x in bottom_20]
        
        plt.bar(range(20), mae_values, color='red', alpha=0.7)
        plt.xticks(range(20), cross_names, rotation=45, ha='right')
        plt.xlabel('Intersection ID')
        plt.ylabel('MAE')
        plt.title('Bottom 20 Worst Performing Intersections')
        plt.grid(True, alpha=0.3)
        
        # 3. R² distribution
        plt.subplot(2, 2, 3)
        r2_values = [x['r2'] for x in intersection_metrics]
        plt.hist(r2_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(np.mean(r2_values), color='red', linestyle='--', 
                   label=f'Mean R²: {np.mean(r2_values):.3f}')
        plt.xlabel('R² Score')
        plt.ylabel('Number of Intersections')
        plt.title('Distribution of R² Scores Across Intersections')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Performance summary
        plt.subplot(2, 2, 4)
        summary_text = f"""
        Intersection Performance Summary
        
        Overall Statistics:
        - Total Intersections: {len(intersection_metrics)}
        - Mean MAE: {np.mean([x['mae'] for x in intersection_metrics]):.2f}
        - Mean RMSE: {np.mean([x['rmse'] for x in intersection_metrics]):.2f}
        - Mean R²: {np.mean(r2_values):.3f}
        
        Best Intersection:
        - ID: {intersection_metrics[0]['cross_id']}
        - MAE: {intersection_metrics[0]['mae']:.2f}
        - R²: {intersection_metrics[0]['r2']:.3f}
        
        Worst Intersection:
        - ID: {intersection_metrics[-1]['cross_id']}
        - MAE: {intersection_metrics[-1]['mae']:.2f}
        - R²: {intersection_metrics[-1]['r2']:.3f}
        
        Performance Distribution:
        - R² > 0.8: {sum(1 for x in r2_values if x > 0.8)} intersections
        - R² > 0.6: {sum(1 for x in r2_values if x > 0.6)} intersections
        - R² < 0.4: {sum(1 for x in r2_values if x < 0.4)} intersections
        """
        
        plt.text(0.05, 0.5, summary_text, fontsize=10, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        plt.axis('off')
        
        plt.suptitle(f'{model_name} - Intersection Performance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folders['comparison'], f'{model_name}_intersection_performance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_peak_hour_analysis(self, y_true, y_pred, model_name):
        """피크/비피크 시간대 분석"""
        plt.figure(figsize=(16, 10))
        
        # Assuming 15-minute intervals, 96 intervals per day
        intervals_per_day = 96
        days = y_true.shape[0] // intervals_per_day
        
        if days > 0:
            # Define peak hours (7-9 AM: 28-36, 5-7 PM: 68-76)
            morning_peak = list(range(28, 36))
            evening_peak = list(range(68, 76))
            peak_intervals = morning_peak + evening_peak
            
            # Calculate hourly performance
            hourly_mae = []
            hourly_volume = []
            
            for hour in range(24):
                hour_intervals = [hour * 4 + i for i in range(4)]
                hour_data = []
                
                for day in range(min(days, 7)):  # Use first week
                    day_start = day * intervals_per_day
                    for interval in hour_intervals:
                        if day_start + interval < y_true.shape[0]:
                            hour_true = y_true[day_start + interval].flatten()
                            hour_pred = y_pred[day_start + interval].flatten()
                            hour_mae_val = mean_absolute_error(hour_true, hour_pred)
                            hour_data.append(hour_mae_val)
                
                if hour_data:
                    hourly_mae.append(np.mean(hour_data))
                    hourly_volume.append(np.mean(y_true[day_start + hour * 4:day_start + (hour + 1) * 4].sum()))
                else:
                    hourly_mae.append(0)
                    hourly_volume.append(0)
            
            # 1. Hourly MAE
            plt.subplot(2, 2, 1)
            bars = plt.bar(range(24), hourly_mae, color='lightblue', alpha=0.7)
            
            # Highlight peak hours
            for hour in [7, 8, 17, 18]:
                if hour < len(bars):
                    bars[hour].set_color('red')
            
            plt.xlabel('Hour of Day')
            plt.ylabel('Average MAE')
            plt.title('Prediction Accuracy by Hour')
            plt.xticks(range(0, 24, 2))
            plt.grid(True, alpha=0.3)
            
            # 2. Traffic volume pattern
            plt.subplot(2, 2, 2)
            plt.plot(range(24), hourly_volume, 'b-o', linewidth=2, markersize=6)
            plt.fill_between([7, 9], [0, 0], [max(hourly_volume), max(hourly_volume)], 
                           alpha=0.2, color='red', label='Morning Peak')
            plt.fill_between([17, 19], [0, 0], [max(hourly_volume), max(hourly_volume)], 
                           alpha=0.2, color='orange', label='Evening Peak')
            plt.xlabel('Hour of Day')
            plt.ylabel('Average Traffic Volume')
            plt.title('Daily Traffic Pattern')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 3. Peak vs Off-peak comparison
            plt.subplot(2, 2, 3)
            
            # Calculate peak and off-peak performance
            peak_mae_vals = []
            off_peak_mae_vals = []
            
            for i in range(min(intervals_per_day, y_true.shape[0])):
                interval_in_day = i % intervals_per_day
                hour = interval_in_day // 4
                
                mae_val = mean_absolute_error(y_true[i].flatten(), y_pred[i].flatten())
                
                if hour in [7, 8, 17, 18]:
                    peak_mae_vals.append(mae_val)
                else:
                    off_peak_mae_vals.append(mae_val)
            
            if peak_mae_vals and off_peak_mae_vals:
                categories = ['Peak Hours', 'Off-Peak Hours']
                mae_means = [np.mean(peak_mae_vals), np.mean(off_peak_mae_vals)]
                mae_stds = [np.std(peak_mae_vals), np.std(off_peak_mae_vals)]
                
                x = np.arange(len(categories))
                plt.bar(x, mae_means, yerr=mae_stds, capsize=10, 
                       color=['red', 'blue'], alpha=0.7)
                plt.xticks(x, categories)
                plt.ylabel('MAE')
                plt.title('Peak vs Off-Peak Performance')
                plt.grid(True, alpha=0.3)
        
        # 4. Summary
        plt.subplot(2, 2, 4)
        summary_text = f"""
        Peak Hour Analysis Summary
        
        Peak Hours:
        - Morning: 7:00-9:00 AM
        - Evening: 5:00-7:00 PM
        
        Performance Comparison:
        - Peak MAE: {np.mean(peak_mae_vals) if 'peak_mae_vals' in locals() and peak_mae_vals else 'N/A':.2f}
        - Off-Peak MAE: {np.mean(off_peak_mae_vals) if 'off_peak_mae_vals' in locals() and off_peak_mae_vals else 'N/A':.2f}
        
        Hourly Performance:
        - Best Hour: {np.argmin(hourly_mae) if 'hourly_mae' in locals() else 'N/A'}:00 
          (MAE: {min(hourly_mae) if 'hourly_mae' in locals() and hourly_mae else 'N/A':.2f})
        - Worst Hour: {np.argmax(hourly_mae) if 'hourly_mae' in locals() else 'N/A'}:00 
          (MAE: {max(hourly_mae) if 'hourly_mae' in locals() and hourly_mae else 'N/A':.2f})
        """
        
        plt.text(0.05, 0.5, summary_text, fontsize=11, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        plt.axis('off')
        
        plt.suptitle(f'{model_name} - Peak Hour Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.folders['comparison'], f'{model_name}_peak_hour_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to generate visualizations from saved models"""
    print("🎨 " + "="*70)
    print("🎨 Traffic GNN Model Visualization from Saved Models")
    print("🎨 " + "="*70)
    
    # Initialize visualizer
    visualizer = ModelVisualizer()
    
    # Check for saved model files
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
    for model_type, model_path in available_models.items():
        print(f"\n{'='*70}")
        print(f"Processing {model_type} Model")
        print(f"{'='*70}")
        
        try:
            # Load model and data
            model, scaler, edge_index, sequences, targets, timestamps, cross_ids = \
                visualizer.load_model_and_data(model_path, model_type)
            
            # Generate predictions
            y_true, y_pred, metrics = visualizer.generate_predictions(
                model, scaler, edge_index, sequences, targets, sample_size=500
            )
            
            # Create all visualizations
            visualizer.create_all_visualizations(y_true, y_pred, metrics, model_type, cross_ids)
            
        except Exception as e:
            print(f"❌ Error processing {model_type}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✅ All visualizations saved in: {visualizer.output_folder}/")
    print("🎉 Visualization generation completed!")

if __name__ == "__main__":
    main()