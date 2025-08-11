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
warnings.filterwarnings('ignore')

# Import model classes from previous implementation
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

class FutureTrafficPredictor:
    """미래 교통량 예측기"""
    
    def __init__(self, model_path, model_type='STGCN'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model_type = model_type
        
        # Load model and data
        self.load_model_and_data()
        
    def load_model_and_data(self):
        """모델과 필요한 데이터 로드"""
        print(f" Loading {self.model_type} model from {self.model_path}...")
        
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
        
        print(" Model loaded successfully!")
        
        # Load graph structure
        print(" Loading graph structure...")
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
        
        # Load sequences to get last known data
        print(" Loading temporal sequences...")
        with open(os.path.join('gnn_data', 'temporal_sequences.pkl'), 'rb') as f:
            seq_data = pickle.load(f)
        
        self.sequences = seq_data['sequences']
        self.timestamps = seq_data.get('timestamps', [])
        
        print(f" Data loaded successfully!")
        print(f"   Last timestamp: {self.timestamps[-1] if self.timestamps else 'Unknown'}")
    
    def create_time_features(self, timestamp):
        """시간 특징 생성"""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        return [hour, day_of_week, is_weekend]
    
    def predict_future_week(self, start_date=None):
        """미래 일주일 예측"""
        print("\n🔮 Predicting future week traffic...")
        
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
        
        print(f" Prediction period: {start_date} to {start_date + timedelta(days=7)}")
        
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
            
            # Update sequence for next prediction
            # Create next input features (simplified - using predictions as features)
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
            
            # Shift sequence and add new features
            current_sequence = np.concatenate([current_sequence[1:], [next_features]], axis=0)
            
            # Update progress bar
            if step % intervals_per_day == 0:
                day_num = step // intervals_per_day + 1
                pbar.set_description(f"Day {day_num}/7")
            
            pbar.update(1)
        
        pbar.close()
        
        # Convert to numpy array
        predictions_array = np.array(all_predictions)  # Shape: [672, 106, 24]
        
        print(f" Generated {len(all_predictions)} predictions")
        print(f"   Shape: {predictions_array.shape}")
        
        return predictions_array, all_timestamps
    
    def save_predictions_to_excel(self, predictions, timestamps, output_folder='future_predictions'):
        """예측 결과를 Excel로 저장"""
        print("\n Saving predictions to Excel...")
        
        os.makedirs(output_folder, exist_ok=True)
        
        # 1. 전체 원시 데이터 저장
        print("   Creating raw data...")
        all_data = []
        
        for t_idx, timestamp in enumerate(timestamps):
            for i_idx, cross_id in enumerate(self.target_cross_ids):
                for d_idx in range(24):
                    all_data.append({
                        'timestamp': timestamp,
                        'date': timestamp.date(),
                        'time': timestamp.time(),
                        'hour': timestamp.hour,
                        'day_of_week': timestamp.strftime('%A'),
                        'cross_id': cross_id,
                        'direction': f'VOL_{d_idx+1:02d}',
                        'predicted_volume': predictions[t_idx, i_idx, d_idx]
                    })
        
        df_raw = pd.DataFrame(all_data)
        
        # 2. 일별 요약 데이터
        print("   Creating daily summaries...")
        daily_summaries = {}
        
        for day in range(7):
            day_start = day * 96
            day_end = (day + 1) * 96
            day_date = timestamps[day_start].date()
            
            # 일별 데이터 추출
            day_predictions = predictions[day_start:day_end]
            
            # 교차로별 일일 총량
            intersection_daily = []
            for i_idx, cross_id in enumerate(self.target_cross_ids):
                daily_total = np.sum(day_predictions[:, i_idx, :])
                daily_avg = np.mean(day_predictions[:, i_idx, :])
                peak_hour_idx = np.argmax(np.sum(day_predictions[:, i_idx, :], axis=1))
                peak_hour = timestamps[day_start + peak_hour_idx].hour
                
                intersection_daily.append({
                    'cross_id': cross_id,
                    'date': day_date,
                    'total_volume': daily_total,
                    'avg_volume_per_interval': daily_avg,
                    'peak_hour': peak_hour,
                    'peak_volume': np.sum(day_predictions[peak_hour_idx, i_idx, :])
                })
            
            daily_summaries[day_date] = pd.DataFrame(intersection_daily)
        
        # 3. 시간대별 평균 (일주일 전체)
        print("   Creating hourly patterns...")
        hourly_patterns = []
        
        for hour in range(24):
            hour_indices = [i for i, ts in enumerate(timestamps) if ts.hour == hour]
            hour_data = predictions[hour_indices]
            
            hourly_patterns.append({
                'hour': hour,
                'avg_total_volume': np.mean(np.sum(hour_data, axis=(1, 2))),
                'avg_intersection_volume': np.mean(np.sum(hour_data, axis=2)),
                'busiest_direction': f'VOL_{np.argmax(np.mean(hour_data, axis=(0, 1))) + 1:02d}',
                'busiest_intersection': self.target_cross_ids[np.argmax(np.mean(np.sum(hour_data, axis=2), axis=0))]
            })
        
        df_hourly = pd.DataFrame(hourly_patterns)
        
        # 4. 방향별 주간 총량
        print("   Creating direction summaries...")
        direction_summary = []
        
        for d_idx in range(24):
            direction_total = np.sum(predictions[:, :, d_idx])
            direction_avg = np.mean(predictions[:, :, d_idx])
            
            direction_summary.append({
                'direction': f'VOL_{d_idx+1:02d}',
                'week_total': direction_total,
                'daily_average': direction_total / 7,
                'interval_average': direction_avg,
                'busiest_hour': timestamps[np.argmax(np.sum(predictions[:, :, d_idx], axis=1))].hour
            })
        
        df_directions = pd.DataFrame(direction_summary)
        
        # 5. Excel 파일로 저장
        excel_path = os.path.join(output_folder, f'future_traffic_predictions_{self.model_type}.xlsx')
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Overview sheet
            overview_data = {
                'Metric': ['Model Type', 'Prediction Start', 'Prediction End', 'Total Days', 
                          'Total Intervals', 'Total Intersections', 'Total Directions'],
                'Value': [self.model_type, timestamps[0], timestamps[-1], 7, 
                         len(timestamps), len(self.target_cross_ids), 24]
            }
            df_overview = pd.DataFrame(overview_data)
            df_overview.to_excel(writer, sheet_name='Overview', index=False)
            
            # Raw predictions (sampled - too large for full data)
            print("   Saving sampled raw data...")
            df_raw_sample = df_raw[df_raw['hour'].isin([0, 6, 12, 18])]  # Sample 4 hours per day
            df_raw_sample.to_excel(writer, sheet_name='Raw_Data_Sample', index=False)
            
            # Daily summaries
            for date, df_daily in daily_summaries.items():
                sheet_name = f'Day_{date.strftime("%Y%m%d")}'
                df_daily.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Hourly patterns
            df_hourly.to_excel(writer, sheet_name='Hourly_Patterns', index=False)
            
            # Direction summary
            df_directions.to_excel(writer, sheet_name='Direction_Summary', index=False)
            
            # Weekly summary by intersection
            print("   Creating weekly intersection summary...")
            intersection_summary = []
            for i_idx, cross_id in enumerate(self.target_cross_ids):
                week_total = np.sum(predictions[:, i_idx, :])
                week_avg = np.mean(predictions[:, i_idx, :])
                busiest_time_idx = np.argmax(np.sum(predictions[:, i_idx, :], axis=1))
                
                intersection_summary.append({
                    'cross_id': cross_id,
                    'week_total_volume': week_total,
                    'daily_average': week_total / 7,
                    'interval_average': week_avg,
                    'busiest_time': timestamps[busiest_time_idx],
                    'busiest_direction': f'VOL_{np.argmax(np.sum(predictions[:, i_idx, :], axis=0)) + 1:02d}'
                })
            
            df_intersection_weekly = pd.DataFrame(intersection_summary)
            df_intersection_weekly.to_excel(writer, sheet_name='Weekly_Intersection_Summary', index=False)
        
        print(f" Predictions saved to: {excel_path}")
        
        # 6. 일별 CSV 파일 저장 (상세 데이터)
        print("\n Saving daily CSV files...")
        csv_folder = os.path.join(output_folder, 'daily_csv')
        os.makedirs(csv_folder, exist_ok=True)
        
        for day in range(7):
            day_start = day * 96
            day_end = (day + 1) * 96
            day_date = timestamps[day_start].date()
            
            day_data = []
            for t_idx in range(day_start, day_end):
                for i_idx, cross_id in enumerate(self.target_cross_ids):
                    row = {
                        'timestamp': timestamps[t_idx],
                        'cross_id': cross_id
                    }
                    # Add all 24 directions
                    for d_idx in range(24):
                        row[f'VOL_{d_idx+1:02d}'] = predictions[t_idx, i_idx, d_idx]
                    
                    # Add total
                    row['TOTAL'] = np.sum(predictions[t_idx, i_idx, :])
                    day_data.append(row)
            
            df_day = pd.DataFrame(day_data)
            csv_path = os.path.join(csv_folder, f'predictions_{day_date.strftime("%Y%m%d")}.csv')
            df_day.to_csv(csv_path, index=False)
            print(f"   Saved: {csv_path}")
        
        return excel_path

def main():
    """메인 실행 함수"""
    print(" " + "="*70)
    print(" Future Traffic Prediction - One Week Forecast")
    print(" " + "="*70)
    
    # Check for available models
    model_files = {
        'STGCN': 'stgcn_traffic_model.pth',
        'GAT': 'gat_traffic_model.pth'
    }
    
    available_models = {}
    for model_type, model_path in model_files.items():
        if os.path.exists(model_path):
            available_models[model_type] = model_path
            print(f" Found {model_type} model: {model_path}")
        else:
            print(f" {model_type} model not found: {model_path}")
    
    if not available_models:
        print("\n No saved models found! Please train models first.")
        return
    
    # Process each available model
    results = {}
    
    for model_type, model_path in available_models.items():
        print(f"\n{'='*70}")
        print(f"Processing {model_type} Model")
        print(f"{'='*70}")
        
        try:
            # Initialize predictor
            predictor = FutureTrafficPredictor(model_path, model_type)
            
            # Generate future predictions
            predictions, timestamps = predictor.predict_future_week()
            
            # Save to Excel
            excel_path = predictor.save_predictions_to_excel(predictions, timestamps)
            
            results[model_type] = {
                'predictions': predictions,
                'timestamps': timestamps,
                'excel_path': excel_path
            }
            
            print(f"\n {model_type} prediction completed!")
            
        except Exception as e:
            print(f"\n Error processing {model_type}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print("\n" + "="*70)
    print(" PREDICTION SUMMARY")
    print("="*70)
    
    for model_type, result in results.items():
        print(f"\n{model_type}:")
        print(f"  - Prediction period: {result['timestamps'][0]} to {result['timestamps'][-1]}")
        print(f"  - Total predictions: {len(result['timestamps']):,} intervals")
        print(f"  - Output file: {result['excel_path']}")
    
    print("\n Future prediction completed!")
    print(" Check 'future_predictions' folder for results")

if __name__ == "__main__":
    main()