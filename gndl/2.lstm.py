import os
import json
import time
import torch
import pickle
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ============= MultiNodeLSTM Model =============

class MultiNodeLSTM(nn.Module):
    """
    LSTM model that processes each intersection separately
    """
    def __init__(self, num_nodes, num_features, hidden_dim=64, lstm_layers=2, 
                 num_directions=24, dropout=0.2):
        super(MultiNodeLSTM, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_directions = num_directions
        self.lstm_layers = lstm_layers
        
        # Separate LSTM for each node (intersection)
        self.node_lstms = nn.ModuleList([
            nn.LSTM(num_features, hidden_dim, lstm_layers, 
                   batch_first=True, dropout=dropout if lstm_layers > 1 else 0)
            for _ in range(num_nodes)
        ])
        
        # Output layer for each node
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_directions)
            )
            for _ in range(num_nodes)
        ])
    
    def forward(self, x):
        """
        x shape: (batch_size, sequence_length, num_nodes, num_features)
        """
        batch_size = x.size(0)
        predictions = []
        
        # Process each node separately
        for node_idx in range(self.num_nodes):
            # Extract data for this node
            node_data = x[:, :, node_idx, :]  # (batch_size, seq_len, num_features)
            
            # LSTM forward pass
            lstm_out, _ = self.node_lstms[node_idx](node_data)
            
            # Take last time step
            last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
            
            # Generate predictions for this node
            node_pred = self.output_layers[node_idx](last_output)
            predictions.append(node_pred)
        
        # Stack predictions: (batch_size, num_nodes, num_directions)
        output = torch.stack(predictions, dim=1)
        
        return output


# ============= LSTM Trainer Class =============

class TrafficLSTMTrainer:
    """
    Trainer for MultiNodeLSTM traffic model
    """
    def __init__(self, data_path='gnn_data', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.data_path = data_path
        
        # Load data
        self.load_data()
        
        # Create model
        self.model = MultiNodeLSTM(
            num_nodes=106,
            num_features=43,
            hidden_dim=64,
            lstm_layers=2,
            num_directions=24,
            dropout=0.2
        ).to(device)
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        print(f"🤖 MultiNodeLSTM model created")
        print(f"📊 Device: {device}")
        print(f"🔧 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def load_data(self):
        """Load preprocessed data"""
        print("📂 Loading preprocessed data...")
        
        # Load temporal sequences
        print("  📈 Loading temporal sequences...")
        with open(os.path.join(self.data_path, 'temporal_sequences.pkl'), 'rb') as f:
            seq_data = pickle.load(f)
        
        self.sequences = seq_data['sequences']
        self.targets = seq_data['targets']
        self.timestamps = seq_data.get('timestamps', None)
        
        # Load graph structure for metadata
        print("  🗺️ Loading metadata...")
        with open(os.path.join(self.data_path, 'graph_structure.pkl'), 'rb') as f:
            graph_data = pickle.load(f)
        
        self.target_cross_ids = graph_data['target_cross_ids']
        
        # Load metadata
        with open(os.path.join(self.data_path, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        
        print(f"✅ Data loaded:")
        print(f"   📊 Total sequences: {self.sequences.shape}")
        print(f"   🎯 Targets: {self.targets.shape}")
        print(f"   🚦 Intersections: {len(self.target_cross_ids)}")
    
    def prepare_data_splits(self, train_ratio=0.7, val_ratio=0.15):
        """Prepare data for LSTM training - 전체 데이터 사용"""
        print("🔄 Preparing data splits (using full dataset)...")
        
        n_samples = len(self.sequences)
        print(f"📊 Total samples: {n_samples:,}")
        
        # Calculate split indices
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        # Split data
        X_train = self.sequences[:train_end]
        X_val = self.sequences[train_end:val_end]
        X_test = self.sequences[val_end:]
        
        y_train = self.targets[:train_end]
        y_val = self.targets[train_end:val_end]
        y_test = self.targets[val_end:]
        
        # Normalize features
        print("  📏 Normalizing features (vectorized)...")
        # per-feature 통계: (sample, seq, node) 축 전체에 대해 평균/표준편차
        self.feature_mean = X_train.mean(axis=(0, 1, 2), dtype=np.float64)   # shape: (num_features,)
        self.feature_std  = X_train.std(axis=(0, 1, 2), dtype=np.float64)
        self.feature_std[self.feature_std < 1e-6] = 1.0  # 분산 0 보호

        # 벡터화 변환 (브로드캐스팅)
        X_train = (X_train - self.feature_mean) / self.feature_std
        X_val   = (X_val   - self.feature_mean) / self.feature_std
        X_test  = (X_test  - self.feature_mean) / self.feature_std
        
        # Convert to tensors
        print("  🧮 Converting to tensors...")
        self.X_train = torch.FloatTensor(X_train).to(self.device)
        self.X_val = torch.FloatTensor(X_val).to(self.device)
        self.X_test = torch.FloatTensor(X_test).to(self.device)
        
        self.y_train = torch.FloatTensor(y_train).to(self.device)
        self.y_val = torch.FloatTensor(y_val).to(self.device)
        self.y_test = torch.FloatTensor(y_test).to(self.device)
        
        print(f"✅ Data splits prepared:")
        print(f"   🚂 Train: {len(X_train):,} samples ({train_ratio*100:.0f}%)")
        print(f"   ✅ Val: {len(X_val):,} samples ({val_ratio*100:.0f}%)")
        print(f"   🧪 Test: {len(X_test):,} samples ({(1-train_ratio-val_ratio)*100:.0f}%)")
    
    def _normalize_sequences(self, sequences):
        normalized = np.zeros_like(sequences)
        for i in tqdm(range(len(sequences)), desc="Normalizing", leave=False):
            seq_flat = sequences[i].reshape(-1, sequences[i].shape[-1])
            seq_normalized = self.scaler.transform(seq_flat)
            normalized[i] = seq_normalized.reshape(sequences[i].shape)
        return (sequences - self.feature_mean) / self.feature_std
    
    def train(self, epochs=100, batch_size=32, learning_rate=0.001):
        """Train MultiNodeLSTM model"""
        print(f"🚀 Starting MultiNodeLSTM training for {epochs} epochs...")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        best_metrics = None
        
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 20
        
        epoch_progress = tqdm(range(epochs), desc="🏃 Training", unit="epoch")
        
        for epoch in epoch_progress:
            # Training phase
            self.model.train()
            total_train_loss = 0
            num_batches = (len(self.X_train) + batch_size - 1) // batch_size
            
            for i in range(0, len(self.X_train), batch_size):
                batch_X = self.X_train[i:i+batch_size]
                batch_y = self.y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / num_batches
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(self.X_val)
                val_loss = criterion(val_predictions, self.y_val).item()
            
            scheduler.step(val_loss)
            
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            
            # Early stopping and best model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                self.best_model_state = self.model.state_dict().copy()
                
                # Calculate and save best metrics
                with torch.no_grad():
                    test_predictions = self.model(self.X_test)
                    y_true = self.y_test.cpu().numpy()
                    y_pred = test_predictions.cpu().numpy()
                    
                    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
                    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_true.flatten(), y_pred.flatten())
                    
                    best_metrics = {
                        'mae': mae,
                        'mse': mse,
                        'rmse': rmse,
                        'r2': r2
                    }
            else:
                patience_counter += 1
            
            # Update progress
            epoch_progress.set_postfix({
                "Train Loss": f"{avg_train_loss:.4f}",
                "Val Loss": f"{val_loss:.4f}",
                "Best": f"{best_val_loss:.4f}",
                "R²": f"{best_metrics['r2']:.4f}" if best_metrics else "N/A",
                "Patience": f"{patience_counter}/{early_stopping_patience}"
            })
            
            if patience_counter >= early_stopping_patience:
                print(f"\n⏹️ Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        print("✅ Training completed!")
        self.best_metrics = best_metrics
        return train_losses, val_losses
    
    def evaluate(self):
        """Evaluate model on test set"""
        print("🧪 Evaluating MultiNodeLSTM model on test set...")
        
        self.model.eval()
        with torch.no_grad():
            test_predictions = self.model(self.X_test)
        
        # Convert to numpy
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
        
        metrics = {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}
        return metrics, y_true, y_pred
    
    def save_model(self, path='multinode_lstm_model.pth'):
        """Save trained model with metrics"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'metadata': self.metadata,
            'target_cross_ids': self.target_cross_ids,
            'model_config': {
                'num_nodes': self.model.num_nodes,
                'num_features': self.model.num_features,
                'hidden_dim': self.model.hidden_dim,
                'lstm_layers': self.model.lstm_layers,
                'num_directions': self.model.num_directions
            }
        }
        
        # Add best metrics if available
        if hasattr(self, 'best_metrics') and self.best_metrics is not None:
            save_dict['best_metrics'] = self.best_metrics
        
        torch.save(save_dict, path)
        print(f"💾 Model saved to {path}")


# ============= Visualization Functions =============

def plot_training_history(train_losses, val_losses):
    """Plot training history"""
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MultiNodeLSTM - Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multinode_lstm_training_history.png', dpi=300)
    plt.close()

def plot_prediction_comparison(y_true, y_pred):
    """Plot prediction vs actual comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Scatter plot
    ax = axes[0, 0]
    sample_size = min(10000, y_true.size)
    idx = np.random.choice(y_true.size, sample_size, replace=False)
    y_true_sample = y_true.flatten()[idx]
    y_pred_sample = y_pred.flatten()[idx]
    
    ax.scatter(y_true_sample, y_pred_sample, alpha=0.5, s=10)
    ax.plot([0, y_true_sample.max()], [0, y_true_sample.max()], 'r--', linewidth=2)
    ax.set_xlabel('Actual Traffic Volume')
    ax.set_ylabel('Predicted Traffic Volume')
    ax.set_title('Predictions vs Actual')
    ax.grid(True, alpha=0.3)
    
    # 2. Time series sample
    ax = axes[0, 1]
    sample_intersection = 0
    sample_direction = 0
    time_steps = min(500, y_true.shape[0])
    
    actual = y_true[:time_steps, sample_intersection, sample_direction]
    predicted = y_pred[:time_steps, sample_intersection, sample_direction]
    
    ax.plot(actual, 'b-', label='Actual', linewidth=2, alpha=0.7)
    ax.plot(predicted, 'r--', label='Predicted', linewidth=2, alpha=0.7)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Traffic Volume')
    ax.set_title('Time Series Comparison (Sample)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Error distribution
    ax = axes[1, 0]
    errors = y_pred - y_true
    ax.hist(errors.flatten(), bins=100, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    ax.text(0.02, 0.98, f'Mean: {np.mean(errors):.2f}\nStd: {np.std(errors):.2f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Performance by direction
    ax = axes[1, 1]
    direction_mae = []
    for i in range(24):
        mae = mean_absolute_error(y_true[:, :, i].flatten(), y_pred[:, :, i].flatten())
        direction_mae.append(mae)
    
    bars = ax.bar(range(24), direction_mae, alpha=0.7)
    # Highlight best and worst
    best_idx = np.argmin(direction_mae)
    worst_idx = np.argmax(direction_mae)
    bars[best_idx].set_color('green')
    bars[worst_idx].set_color('red')
    
    ax.set_xlabel('Direction (VOL_XX)')
    ax.set_ylabel('MAE')
    ax.set_title('Performance by Direction')
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('MultiNodeLSTM - Prediction Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig('multinode_lstm_prediction_analysis.png', dpi=300)
    plt.close()

# ============= Main Function =============

def main():
    """Main execution function for MultiNodeLSTM traffic prediction"""
    print("🚦 " + "="*70)
    print("🚦 Traffic MultiNodeLSTM Model Training Pipeline")
    print("🚦 Using Full Dataset")
    print("🚦 " + "="*70)
    
    # Check if preprocessed data exists
    if not os.path.exists('gnn_data'):
        print("❌ Preprocessed data not found. Please run 1.preprocess.py first.")
        return
    
    try:
        # Initialize trainer
        trainer = TrafficLSTMTrainer()
        
        # Prepare data (full dataset)
        trainer.prepare_data_splits(
            train_ratio=0.7,
            val_ratio=0.15
        )
        
        # Train model
        start_time = time.time()
        train_losses, val_losses = trainer.train(
            epochs=100,  # More epochs since using full data
            batch_size=32,
            learning_rate=0.001
        )
        training_time = time.time() - start_time
        
        # Evaluate
        metrics, y_true, y_pred = trainer.evaluate()
        
        # Save model
        model_path = 'multinode_lstm_traffic_model.pth'
        trainer.save_model(model_path)
        
        # Create visualizations
        print(f"\n📊 Creating visualizations...")
        plot_training_history(train_losses, val_losses)
        plot_prediction_comparison(y_true, y_pred)
        
        # Final summary
        print("\n" + "="*70)
        print("📊 TRAINING SUMMARY")
        print("="*70)
        print(f"Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        print(f"Final Performance:")
        print(f"  - MAE: {metrics['mae']:.4f}")
        print(f"  - RMSE: {metrics['rmse']:.4f}")
        print(f"  - R²: {metrics['r2']:.4f}")
        print(f"  - Accuracy: {metrics['r2']*100:.1f}%")
        print(f"\nModel saved: {model_path}")
        print("\n🎉 MultiNodeLSTM training completed!")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()