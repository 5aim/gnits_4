import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from tqdm import tqdm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ============= GNN Models =============

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

# ============= Trainer Class =============

class TrafficGNNTrainer:
    """
    Trainer class for Traffic GNN models
    """
    def __init__(self, model, data_path='gnn_data', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.data_path = data_path
        
        # Load processed data
        self.load_data()
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        print(f"🤖 Model loaded on {device}")
        
    def load_data(self):
        """Load preprocessed data"""
        print("📂 Loading preprocessed data...")
        
        # Load graph structure
        print("  🗺️  Loading graph structure...")
        with open(os.path.join(self.data_path, 'graph_structure.pkl'), 'rb') as f:
            graph_data = pickle.load(f)
        
        self.edges = graph_data['edges']
        
        # Create edge_index with proper node ID mapping
        self.target_cross_ids = graph_data['target_cross_ids']
        
        # Create mapping from cross_id to index (0 to 105)
        cross_id_to_idx = {cross_id: idx for idx, cross_id in enumerate(self.target_cross_ids)}
        
        # Convert edges to use indices instead of cross_ids
        edge_list = []
        for edge in self.edges:
            source_idx = cross_id_to_idx.get(edge[0])
            target_idx = cross_id_to_idx.get(edge[1])
            if source_idx is not None and target_idx is not None:
                edge_list.append([source_idx, target_idx])
        
        if edge_list:
            self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            self.edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Load temporal sequences
        print("  📈 Loading temporal sequences...")
        with open(os.path.join(self.data_path, 'temporal_sequences.pkl'), 'rb') as f:
            seq_data = pickle.load(f)
        
        self.sequences = seq_data['sequences']
        self.targets = seq_data['targets']
        self.timestamps = seq_data.get('timestamps', None)
        
        # Load metadata
        print("  📋 Loading metadata...")
        with open(os.path.join(self.data_path, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        
        print(f"✅ Data loaded:")
        print(f"   📊 Sequences: {len(self.sequences):,}")
        print(f"   📐 Sequence shape: {self.sequences.shape}")
        print(f"   🎯 Target shape: {self.targets.shape}")
        print(f"   🔗 Graph edges: {self.edge_index.size(1)}")
        print(f"   📍 Node indices range: 0 to {len(self.target_cross_ids)-1}")
        
    def prepare_data_splits(self, train_ratio=0.7, val_ratio=0.15, max_samples=None):
        """Split data into train/validation/test sets"""
        print("🔄 Preparing data splits...")
        
        n_samples = len(self.sequences)
        
        # Limit data size if specified
        if max_samples and n_samples > max_samples:
            print(f"  📊 Reducing dataset: {n_samples:,} → {max_samples:,} samples")
            indices = np.linspace(0, n_samples-1, max_samples, dtype=int)
            self.sequences = self.sequences[indices]
            self.targets = self.targets[indices]
            n_samples = max_samples
        
        # Calculate split indices
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        # Split sequences
        X_train = self.sequences[:train_end]
        X_val = self.sequences[train_end:val_end]
        X_test = self.sequences[val_end:]
        
        y_train = self.targets[:train_end]
        y_val = self.targets[train_end:val_end]
        y_test = self.targets[val_end:]
        
        # Normalize features
        print("  📏 Normalizing features...")
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        self.scaler.fit(X_train_flat)
        
        # Apply normalization
        print("  🔄 Applying normalization...")
        X_train = self._normalize_sequences(X_train)
        X_val = self._normalize_sequences(X_val)
        X_test = self._normalize_sequences(X_test)
        
        # Convert to tensors
        print("  🧮 Converting to tensors...")
        self.X_train = torch.FloatTensor(X_train).to(self.device)
        self.X_val = torch.FloatTensor(X_val).to(self.device)
        self.X_test = torch.FloatTensor(X_test).to(self.device)
        
        self.y_train = torch.FloatTensor(y_train).to(self.device)
        self.y_val = torch.FloatTensor(y_val).to(self.device)
        self.y_test = torch.FloatTensor(y_test).to(self.device)
        
        self.edge_index = self.edge_index.to(self.device)
        
        print(f"✅ Data splits prepared:")
        print(f"   🚂 Train: {len(X_train):,} samples")
        print(f"   ✅ Validation: {len(X_val):,} samples")
        print(f"   🧪 Test: {len(X_test):,} samples")
        
    def _normalize_sequences(self, sequences):
        """Normalize sequence data"""
        normalized = np.zeros_like(sequences)
        for i in tqdm(range(len(sequences)), desc="Normalizing", leave=False):
            seq_flat = sequences[i].reshape(-1, sequences[i].shape[-1])
            seq_normalized = self.scaler.transform(seq_flat)
            normalized[i] = seq_normalized.reshape(sequences[i].shape)
        return normalized
    
    def train(self, epochs=100, batch_size=32, learning_rate=0.001):
        """Train the model"""
        print(f"🚀 Starting training for {epochs} epochs...")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        # Training progress bar
        epoch_progress = tqdm(range(epochs), desc="🏃 Training", unit="epoch", colour="blue")
        
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 20
        
        for epoch in epoch_progress:
            # Training
            self.model.train()
            total_train_loss = 0
            num_batches = (len(self.X_train) + batch_size - 1) // batch_size
            
            # Batch progress
            batch_progress = tqdm(range(0, len(self.X_train), batch_size), 
                                desc=f"Epoch {epoch+1}", leave=False, unit="batch", colour="green")
            
            for i in batch_progress:
                batch_X = self.X_train[i:i+batch_size]
                batch_y = self.y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(batch_X, self.edge_index)
                loss = criterion(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_train_loss += loss.item()
                
                # Update batch progress
                batch_progress.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "Batch": f"{(i//batch_size)+1}/{num_batches}"
                })
            
            batch_progress.close()
            
            avg_train_loss = total_train_loss / num_batches
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(self.X_val, self.edge_index)
                val_loss = criterion(val_predictions, self.y_val).item()
            
            scheduler.step(val_loss)
            
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Update epoch progress
            epoch_progress.set_postfix({
                "Train Loss": f"{avg_train_loss:.4f}",
                "Val Loss": f"{val_loss:.4f}",
                "Best": f"{best_val_loss:.4f}",
                "Patience": f"{patience_counter}/{early_stopping_patience}"
            })
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                break
        
        epoch_progress.close()
        print("✅ Training completed!")
        return train_losses, val_losses
    
    def evaluate(self):
        """Evaluate model on test set"""
        print("🧪 Evaluating model on test set...")
        
        self.model.eval()
        with torch.no_grad():
            test_predictions = self.model(self.X_test, self.edge_index)
        
        # Convert to numpy for metrics
        y_true = self.y_test.cpu().numpy()
        y_pred = test_predictions.cpu().numpy()
        
        # Calculate metrics
        mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
        mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true.flatten(), y_pred.flatten())
        
        print(f"📊 Test Results:")
        print(f"   MAE:  {mae:.4f}")
        print(f"   MSE:  {mse:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   R²:   {r2:.4f}")
        
        return {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}, y_true, y_pred
    
    def save_model(self, path='traffic_gnn_model.pth'):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'metadata': self.metadata
        }, path)
        print(f"💾 Model saved to {path}")

# ============= Analysis Classes =============

class ComprehensiveAnalyzer:
    """종합 분석 및 시각화 클래스"""
    
    def __init__(self, output_folder='analysis_results'):
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        
        # 서브 폴더 생성
        self.folders = {
            'quantitative': os.path.join(output_folder, 'quantitative_analysis'),
            'visual': os.path.join(output_folder, 'visual_analysis'),
            'excel': os.path.join(output_folder, 'excel_outputs'),
            'comparison': os.path.join(output_folder, 'comparison_analysis')
        }
        
        for folder in self.folders.values():
            os.makedirs(folder, exist_ok=True)
        
        print(f"📁 Analysis folders created in {output_folder}/")
    
    def quantitative_accuracy_analysis(self, y_true, y_pred, model_name):
        """딥러닝 모델링 정확도 정량적 분석"""
        print(f"\n📊 Quantitative Accuracy Analysis - {model_name}")
        print("="*60)
        
        # 전체 예측 정확도
        mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
        mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true.flatten(), y_pred.flatten())
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # 방향별 정확도
        direction_metrics = {}
        for i in range(24):
            dir_true = y_true[:, :, i].flatten()
            dir_pred = y_pred[:, :, i].flatten()
            direction_metrics[f'VOL_{i+1:02d}'] = {
                'MAE': mean_absolute_error(dir_true, dir_pred),
                'RMSE': np.sqrt(mean_squared_error(dir_true, dir_pred)),
                'MAPE': np.mean(np.abs((dir_true - dir_pred) / (dir_true + 1e-8))) * 100
            }
        
        # 교차로별 정확도
        intersection_metrics = {}
        for i in range(y_true.shape[1]):
            int_true = y_true[:, i, :].flatten()
            int_pred = y_pred[:, i, :].flatten()
            intersection_metrics[f'Intersection_{i}'] = {
                'MAE': mean_absolute_error(int_true, int_pred),
                'RMSE': np.sqrt(mean_squared_error(int_true, int_pred)),
                'R2': r2_score(int_true, int_pred)
            }
        
        # 결과 저장
        results = {
            'overall': {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape
            },
            'direction_wise': direction_metrics,
            'intersection_wise': intersection_metrics
        }
        
        # 텍스트 파일로 저장
        report_path = os.path.join(self.folders['quantitative'], f'{model_name}_accuracy_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"{'='*60}\n")
            f.write(f"Quantitative Accuracy Analysis - {model_name}\n")
            f.write(f"{'='*60}\n\n")
            
            f.write("1. Overall Metrics:\n")
            f.write(f"   - MAE:  {mae:.4f}\n")
            f.write(f"   - MSE:  {mse:.4f}\n")
            f.write(f"   - RMSE: {rmse:.4f}\n")
            f.write(f"   - R²:   {r2:.4f}\n")
            f.write(f"   - MAPE: {mape:.2f}%\n\n")
            
            f.write("2. Top 5 Best Performing Directions:\n")
            sorted_dirs = sorted(direction_metrics.items(), key=lambda x: x[1]['MAE'])
            for i, (dir_name, metrics) in enumerate(sorted_dirs[:5]):
                f.write(f"   {i+1}. {dir_name}: MAE={metrics['MAE']:.4f}, MAPE={metrics['MAPE']:.2f}%\n")
            
            f.write("\n3. Top 5 Worst Performing Directions:\n")
            for i, (dir_name, metrics) in enumerate(sorted_dirs[-5:]):
                f.write(f"   {i+1}. {dir_name}: MAE={metrics['MAE']:.4f}, MAPE={metrics['MAPE']:.2f}%\n")
        
        print(f"✅ Accuracy report saved to {report_path}")
        return results
    
    def save_predictions_to_excel(self, y_true, y_pred, model_name, timestamps, cross_ids):
        """예측값과 실제값을 Excel로 저장"""
        print(f"\n💾 Saving predictions to Excel - {model_name}")
        print("="*60)
        
        # 데이터 준비
        all_data = []
        
        for t in range(min(100, y_true.shape[0])):  # 처음 100개 시점만
            for i in range(y_true.shape[1]):  # 각 교차로
                for d in range(24):  # 각 방향
                    all_data.append({
                        'timestamp': timestamps[t] if timestamps is not None and t < len(timestamps) else f'T{t}',
                        'cross_id': cross_ids[i] if i < len(cross_ids) else f'Cross_{i}',
                        'direction': f'VOL_{d+1:02d}',
                        'actual': y_true[t, i, d],
                        'predicted': y_pred[t, i, d],
                        'error': y_pred[t, i, d] - y_true[t, i, d],
                        'abs_error': abs(y_pred[t, i, d] - y_true[t, i, d]),
                        'relative_error': abs(y_pred[t, i, d] - y_true[t, i, d]) / (y_true[t, i, d] + 1e-8) * 100
                    })
        
        # DataFrame 생성
        df_predictions = pd.DataFrame(all_data)
        
        # Excel 파일로 저장
        excel_path = os.path.join(self.folders['excel'], f'{model_name}_predictions_raw_data.xlsx')
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 원시 데이터
            df_predictions.to_excel(writer, sheet_name='Raw_Data', index=False)
            
            # 시간대별 집계
            if 'timestamp' in df_predictions.columns:
                hourly_summary = df_predictions.groupby('timestamp').agg({
                    'actual': ['sum', 'mean'],
                    'predicted': ['sum', 'mean'],
                    'abs_error': 'mean',
                    'relative_error': 'mean'
                }).round(2)
                hourly_summary.to_excel(writer, sheet_name='Hourly_Summary')
            
            # 교차로별 집계
            intersection_summary = df_predictions.groupby('cross_id').agg({
                'actual': ['sum', 'mean'],
                'predicted': ['sum', 'mean'],
                'abs_error': 'mean',
                'relative_error': 'mean'
            }).round(2)
            intersection_summary.to_excel(writer, sheet_name='Intersection_Summary')
            
            # 방향별 집계
            direction_summary = df_predictions.groupby('direction').agg({
                'actual': ['sum', 'mean'],
                'predicted': ['sum', 'mean'],
                'abs_error': 'mean',
                'relative_error': 'mean'
            }).round(2)
            direction_summary.to_excel(writer, sheet_name='Direction_Summary')
        
        print(f"✅ Predictions saved to {excel_path}")
        return df_predictions

# ============= Main Function =============

def main():
    """메인 실행 함수"""
    print("🚦 " + "="*70)
    print("🚦 Traffic GNN Model Training & Analysis Pipeline")
    print("🚦 " + "="*70)
    
    # Check if preprocessed data exists
    if not os.path.exists('gnn_data'):
        print("❌ Preprocessed data not found. Please run preprocess.py first.")
        return
    
    # Initialize analyzer
    analyzer = ComprehensiveAnalyzer()
    
    # Model configurations
    models_config = {
        'STGCN': {
            'class': SpatioTemporalGCN,
            'params': {'num_features': 43, 'hidden_dim': 64, 'num_layers': 2}  # 43 features based on preprocessing
        },
        'GAT': {
            'class': TrafficGAT,
            'params': {'num_features': 43, 'hidden_dim': 64, 'num_heads': 8, 'num_layers': 3}
        }
    }
    
    results = {}
    
    for model_name, config in models_config.items():
        print(f"\n{'='*70}")
        print(f"🚀 Training {model_name} model...")
        print(f"{'='*70}")
        
        try:
            # Initialize model
            model = config['class'](**config['params'])
            
            # Initialize trainer
            trainer = TrafficGNNTrainer(model, data_path='gnn_data')
            
            # Prepare data splits
            trainer.prepare_data_splits(
                train_ratio=0.7,
                val_ratio=0.15,
                max_samples=5000  # Limit samples for faster training
            )
            
            # Train model
            start_time = time.time()
            train_losses, val_losses = trainer.train(
                epochs=30,
                batch_size=32,
                learning_rate=0.001
            )
            training_time = time.time() - start_time
            
            # Evaluate model
            metrics, y_true, y_pred = trainer.evaluate()
            
            # Save model
            model_path = f'{model_name.lower()}_traffic_model.pth'
            trainer.save_model(model_path)
            
            # Store results
            results[model_name] = {
                'metrics': metrics,
                'training_time': training_time,
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            
            # Perform analysis
            print(f"\n📊 Performing analysis for {model_name}...")
            
            # 1. Quantitative accuracy analysis
            accuracy_results = analyzer.quantitative_accuracy_analysis(y_true, y_pred, model_name)
            
            # 2. Save predictions to Excel
            predictions_df = analyzer.save_predictions_to_excel(
                y_true, y_pred, model_name, 
                trainer.timestamps, trainer.target_cross_ids
            )
            
            print(f"✅ {model_name} analysis completed!")
            
        except Exception as e:
            print(f"❌ Error processing {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    print("\n" + "="*70)
    print("📊 FINAL RESULTS SUMMARY")
    print("="*70)
    
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Training Time: {result['training_time']:.2f} seconds")
        print(f"  MAE: {result['metrics']['mae']:.4f}")
        print(f"  RMSE: {result['metrics']['rmse']:.4f}")
        print(f"  R²: {result['metrics']['r2']:.4f}")
    
    print(f"\n✅ All results saved in: {analyzer.output_folder}/")
    print("🎉 Analysis pipeline completed!")
    
    return results

if __name__ == "__main__":
    results = main()