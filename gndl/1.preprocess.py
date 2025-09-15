import os
import sys
import glob
import json
import time
import pickle
import pyodbc
import warnings
import pandas as pd
import numpy as np

from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

load_dotenv()
FLASK_ENV = os.getenv("FLASK_ENV", "production")

DSNNAME = os.getenv("DSNNAME")
DBUSER = os.getenv("DBUSER")
DBPWD = os.getenv("DBPWD")

ENZERO_SERVER = os.getenv("ENZERO_SERVER")
ENZERO_PORT = os.getenv("ENZERO_PORT")
ENZERO_DB = os.getenv("ENZERO_DB")
ENZERO_UID = os.getenv("ENZERO_UID")
ENZERO_PWD = os.getenv("ENZERO_PWD")

def get_connection():
    if FLASK_ENV == "test":
        print(f">>> [INFO] Flask 환경 설정: {FLASK_ENV} (엔제로 서버 사용)")
        return pyodbc.connect(
            f"DRIVER=Tibero 5 ODBC Driver;"
            f"SERVER={ENZERO_SERVER};"
            f"PORT={ENZERO_PORT};"
            f"DB={ENZERO_DB};"
            f"UID={ENZERO_UID};"
            f"PWD={ENZERO_PWD};"
        )
    else:
        print(f">>> [INFO] Flask 환경 설정: {FLASK_ENV} (센터 DSN 사용)")
        return pyodbc.connect(
            f"DSN={DSNNAME};"
            f"UID={DBUSER};"
            f"PWD={DBPWD}"
        )

def _query_min_max_stat15(conn):
    sql = "SELECT MIN(STAT_15MIN) AS MIN_T, MAX(STAT_15MIN) AS MAX_T FROM TOMMS.STAT_15MIN_CROSS WHERE INFRA_TYPE = 'SMT'"
    df = pd.read_sql(sql, conn)
    return str(df['MIN_T'].iloc[0]), str(df['MAX_T'].iloc[0])

def fetch_15min_from_db(start: str|None = None,
                        end: str|None = None,
                        cross_ids: list[int]|None = None,
                        infra_type: str = "SMT",
                        batch_days: int = 7) -> pd.DataFrame:
    """
    STAT_15MIN_CROSS에서 15분 교통량을 배치로 읽어 공통 스키마로 반환.
    반환 컬럼: STAT_TIME, TIME_INTERVAL('15min'), CROSS_ID, INFRA_TYPE, VOL, VPHG, VS,
              VOL_01..VOL_24, (선택) LOS, WALKER_CNT, CRASH_CNT, DELAY_TIME
    """
    time_col = "STAT_15MIN"
    base_cols = [time_col, "CROSS_ID", "INFRA_TYPE", "VOL", "VPHG", "VS"] + [f"VOL_{i:02d}" for i in range(1,25)]
    optional = ["LOS", "WALKER_CNT", "CRASH_CNT", "DELAY_TIME"]
    select_cols = ", ".join([*base_cols, *optional])

    where = ["INFRA_TYPE = ?"]
    params_base = [infra_type]

    if cross_ids:
        placeholders = ",".join(["?"] * len(cross_ids))
        where.append(f"CROSS_ID IN ({placeholders})")
        params_base += cross_ids

    sql_tpl = f"""
      SELECT {select_cols}
      FROM TOMMS.STAT_15MIN_CROSS
      WHERE {" AND ".join(where + ["{time_col} >= ?", "{time_col} < ?"])}
      ORDER BY {time_col}, CROSS_ID
    """.replace("{time_col}", time_col)

    cn = get_connection()
    try:
        # 전체 조회 요청이면 DB에서 기간 경계 먼저 얻기
        if start is None or end is None:
            min_t, max_t = _query_min_max_stat15(cn)   # e.g. '202407010000', '202508312345'
            start = min_t
            # max_t는 inclusive일 수 있으니 +15분 해서 exclusive로 맞춘다
            dt_max = pd.to_datetime(max_t, format="%Y%m%d%H%M") + pd.Timedelta(minutes=15)
            end = dt_max.strftime("%Y%m%d%H%M")

        cur_start = pd.to_datetime(start, format="%Y%m%d%H%M")
        cur_end   = pd.to_datetime(end,   format="%Y%m%d%H%M")
        step = pd.Timedelta(days=batch_days)

        dfs = []

        # ✅ tqdm 프로그레스바 추가
        total_batches = int((cur_end - cur_start) / step) + 1
        print(f"📊 전체 조회 범위: {start} ~ {end}, 배치 크기: {batch_days}일")
        print(f"📊 총 배치 수: {total_batches}")
        progress = tqdm(total=total_batches, desc="DB Fetch", unit="batch")

        while cur_start < cur_end:
            s = cur_start.strftime("%Y%m%d%H%M")
            e = min(cur_start + step, cur_end).strftime("%Y%m%d%H%M")
            params = params_base + [s, e]

            df = pd.read_sql(sql_tpl, cn, params=params)

            if not df.empty:
                df.rename(columns={time_col: "STAT_TIME"}, inplace=True)
                df["TIME_INTERVAL"] = "15min"
                dfs.append(df)

            # 진행상황 업데이트
            progress.set_postfix({"rows": len(df)})
            progress.update(1)

            cur_start += step

        progress.close()

        data = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        return data
    finally:
        cn.close()

class GNNTrafficDataPreprocessor:
    def __init__(self, data_folder='data', output_folder='gnn_data', sa_cross_file='SA_CROSS.csv'):
        """
        GNN Traffic Data Preprocessor with Real Road Network
        
        Args:
            data_folder (str): Folder containing raw CSV files
            output_folder (str): Output folder for processed GNN data
            sa_cross_file (str): SA_CROSS.csv file path for road network info
        """
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.sa_cross_file = sa_cross_file
        
        # Target intersection IDs (106 intersections)
        self.target_cross_ids = [2,3,7,10,17,19,20,22,29,37,39,41,43,53,57,58,61,63,67,72,77,78,79,81,85,86,89,92,94,95,99,101,105,106,108,110,115,119,125,127,129,131,132,134,135,136,138,143,147,148,173,175,177,178,179,181,182,183,190,193,194,195,196,197,199,201,202,207,209,210,217,222,224,227,229,236,237,248,249,250,251,256,260,266,268,279,280,284,285,286,291,292,295,296,298,299,300,305,900,902,903,904,905,906,907,910]
        
        # Direction mapping (VOL_01~24 to 8 directions)
        self.direction_mapping = {
            'S': [1, 2, 3],      # South
            'E': [4, 5, 6],      # East  
            'N': [7, 8, 9],      # North
            'W': [10, 11, 12],   # West
            'SE': [13, 14, 15],  # Southeast
            'NE': [16, 17, 18],  # Northeast
            'NW': [19, 20, 21],  # Northwest
            'SW': [22, 23, 24]   # Southwest
        }
        
        self.direction_names = ['S', 'E', 'N', 'W', 'SE', 'NE', 'NW', 'SW']
        
        # Feature columns for nodes
        self.node_feature_columns = ['VOL', 'VPHG', 'VS'] + [f'VOL_{i:02d}' for i in range(1, 25)]
        
        # Load SA_CROSS information
        self.sa_cross_info = None
        self.load_sa_cross_info()
        
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"🚀 GNN Data Preprocessor Initialized")
        print(f"📍 Target intersections: {len(self.target_cross_ids)}")
        print(f"🗺️  Using real road network from: {sa_cross_file}")
        print(f"📁 Output folder: {output_folder}")
        print("="*60)

    def load_from_db(self, start=None, end=None):
        cn = get_connection()   # 외부 함수 호출
        try:
            sql = f"""
            SELECT STAT_15MIN, CROSS_ID, INFRA_TYPE, VOL, VPHG, VS,
                   {",".join([f"VOL_{i:02d}" for i in range(1,25)])},
                   LOS, WALKER_CNT, CRASH_CNT, DELAY_TIME
            FROM TOMMS.STAT_15MIN_CROSS
            WHERE INFRA_TYPE = 'SMT'
            """
            if start and end:
                sql += f" AND STAT_15MIN BETWEEN '{start}' AND '{end}'"
            df = pd.read_sql(sql, cn)
        finally:
            cn.close()
        
        df.rename(columns={"STAT_15MIN": "STAT_TIME"}, inplace=True)
        df["TIME_INTERVAL"] = "15min"
        return df

    def load_sa_cross_info(self):
        """Load SA_CROSS.csv for real road network connections"""
        if os.path.exists(self.sa_cross_file):
            print(f"📍 Loading road network information from {self.sa_cross_file}")
            self.sa_cross_info = pd.read_csv(self.sa_cross_file)
            
            # Convert CROSS_ID to int if it's float
            if 'CROSS_ID' in self.sa_cross_info.columns:
                self.sa_cross_info['CROSS_ID'] = self.sa_cross_info['CROSS_ID'].fillna(0).astype(int)
            
            print(f"✅ Loaded {len(self.sa_cross_info)} intersection connection records")
            print(f"   - SA Groups: {self.sa_cross_info['SA_ID'].nunique()}")
            print(f"   - Districts: {self.sa_cross_info['DIST'].unique()}")
        else:
            print(f"⚠️  SA_CROSS.csv not found. Will use grid-based connections.")
            self.sa_cross_info = None

    def create_real_road_network(self):
        """Create adjacency list based on real road network from SA_CROSS.csv"""
        if self.sa_cross_info is None:
            print("⚠️  No SA_CROSS info available. Using grid-based network.")
            return self.create_grid_based_network()
        
        print("🗺️  Creating real road network connections...")
        
        adjacency_list = {cross_id: set() for cross_id in self.target_cross_ids}
        
        # Filter SA_CROSS info for target intersections only
        target_sa_info = self.sa_cross_info[self.sa_cross_info['CROSS_ID'].isin(self.target_cross_ids)].copy()
        
        # Group by SA_ID and UPDOWN to find connected intersections
        sa_groups = target_sa_info.groupby(['SA_ID', 'UPDOWN'])
        
        connection_count = 0
        
        # Progress bar for SA groups
        group_progress = tqdm(sa_groups, desc="🔗 Processing SA groups", unit="group")
        
        for (sa_id, updown), group_data in group_progress:
            # Sort by SEQU to get the order of intersections
            group_data = group_data.sort_values('SEQU')
            cross_ids = group_data['CROSS_ID'].tolist()
            
            # Connect consecutive intersections in the same SA group
            for i in range(len(cross_ids) - 1):
                curr_id = cross_ids[i]
                next_id = cross_ids[i + 1]
                
                if curr_id in self.target_cross_ids and next_id in self.target_cross_ids:
                    # Add bidirectional connection
                    adjacency_list[curr_id].add(next_id)
                    adjacency_list[next_id].add(curr_id)
                    connection_count += 1
            
            group_progress.set_postfix({
                "SA_ID": sa_id,
                "Direction": updown,
                "Intersections": len(cross_ids),
                "Total Connections": connection_count
            })
        
        group_progress.close()
        
        # Also add connections between UP and DOWN directions at the same intersection
        # (if they exist in different SA groups)
        print("🔄 Adding UP/DOWN connections...")
        
        for cross_id in tqdm(self.target_cross_ids, desc="Processing UP/DOWN"):
            cross_records = target_sa_info[target_sa_info['CROSS_ID'] == cross_id]
            
            if len(cross_records) > 1:  # Multiple records for same intersection
                sa_ids = cross_records['SA_ID'].unique()
                
                # Find other intersections in the same SA groups
                for sa_id in sa_ids:
                    same_sa = target_sa_info[target_sa_info['SA_ID'] == sa_id]
                    connected_crosses = same_sa['CROSS_ID'].unique()
                    
                    for connected_id in connected_crosses:
                        if connected_id != cross_id and connected_id in self.target_cross_ids:
                            adjacency_list[cross_id].add(connected_id)
                            connection_count += 1
        
        # Convert sets to lists
        adjacency_list = {k: list(v) for k, v in adjacency_list.items()}
        
        # Create edge list
        edges = []
        for source, targets in adjacency_list.items():
            for target in targets:
                if source < target:  # Avoid duplicates
                    edges.append((source, target))
        
        # Print network statistics
        print(f"\n📊 Real Road Network Statistics:")
        print(f"   - Total edges: {len(edges)}")
        print(f"   - Average degree: {sum(len(v) for v in adjacency_list.values()) / len(adjacency_list):.2f}")
        
        # Find isolated nodes
        isolated_nodes = [node for node, neighbors in adjacency_list.items() if len(neighbors) == 0]
        if isolated_nodes:
            print(f"   - ⚠️  Isolated intersections: {len(isolated_nodes)}")
            print(f"        {isolated_nodes[:10]}{'...' if len(isolated_nodes) > 10 else ''}")
        
        # Analyze connectivity by SA group
        print(f"\n🗺️  Connectivity by SA Group:")
        sa_summary = target_sa_info.groupby('SA_ID').agg({
            'CROSS_ID': 'count',
            'DIST': 'first'
        }).sort_values('CROSS_ID', ascending=False)
        
        for sa_id, row in sa_summary.head(5).iterrows():
            print(f"   - SA_{sa_id} ({row['DIST']}): {row['CROSS_ID']} intersections")
        
        print("="*60)
        
        return adjacency_list, edges

    def create_grid_based_network(self):
        """Fallback: Create grid-based network if SA_CROSS not available"""
        print("📐 Creating grid-based network (fallback)...")
        
        adjacency_list = {cross_id: [] for cross_id in self.target_cross_ids}
        
        # Create grid positions
        grid_positions = {}
        cols = 10
        
        for i, cross_id in enumerate(self.target_cross_ids):
            if i < 100:  # First 100 nodes: 10x10 grid
                row = i // cols
                col = i % cols
            else:  # Last 6 nodes: centered in last row
                row = 10
                col = (i - 100) + 2
            
            grid_positions[cross_id] = (row, col)
        
        # Create connections based on grid adjacency
        for cross_id, (row, col) in grid_positions.items():
            # Check 4 neighbors (N, S, E, W)
            neighbors = [
                (row-1, col), (row+1, col), 
                (row, col-1), (row, col+1)
            ]
            
            for n_row, n_col in neighbors:
                # Find cross_id at neighbor position
                for other_id, (other_row, other_col) in grid_positions.items():
                    if (other_row, other_col) == (n_row, n_col):
                        adjacency_list[cross_id].append(other_id)
                        break
        
        # Create edge list
        edges = []
        for source, targets in adjacency_list.items():
            for target in targets:
                if source < target:
                    edges.append((source, target))
        
        print(f"✅ Created grid-based network with {len(edges)} edges")
        return adjacency_list, edges

    def analyze_graph_structure(self, data):
        """Analyze graph structure using real road network"""
        print("🗺️  Analyzing graph structure...")
        
        # Create real road network connections
        adjacency_list, edges = self.create_real_road_network()
        
        # Analyze road directions at each intersection
        intersection_roads = {}
        
        intersection_progress = tqdm(self.target_cross_ids, 
                                   desc="🚦 Analyzing intersection roads", 
                                   unit="intersection")
        
        for cross_id in intersection_progress:
            cross_data = data[data['CROSS_ID'] == cross_id]
            
            if len(cross_data) > 0:
                # Analyze which directions have roads
                roads = {}
                sample_data = cross_data.iloc[0]
                
                for direction, vol_indices in self.direction_mapping.items():
                    has_road = False
                    total_volume = 0
                    
                    for vol_idx in vol_indices:
                        vol_key = f'VOL_{vol_idx:02d}'
                        if vol_key in sample_data:
                            vol_val = sample_data[vol_key]
                            if pd.notna(vol_val) and vol_val > 0:
                                has_road = True
                                # Convert to float if needed
                                if isinstance(vol_val, (int, float)):
                                    total_volume += float(vol_val)
                                else:
                                    total_volume += 0
                    
                    roads[direction] = has_road
                
                intersection_roads[cross_id] = roads
                
                # Add SA group info if available
                if self.sa_cross_info is not None:
                    sa_info = self.sa_cross_info[self.sa_cross_info['CROSS_ID'] == cross_id]
                    if not sa_info.empty:
                        intersection_roads[cross_id]['sa_groups'] = sa_info['SA_ID'].tolist()
                        intersection_roads[cross_id]['district'] = sa_info['DIST'].iloc[0]
                        intersection_roads[cross_id]['name'] = sa_info['CROSS_NAME'].iloc[0] if 'CROSS_NAME' in sa_info.columns else f"Cross_{cross_id}"
            else:
                intersection_roads[cross_id] = {dir: False for dir in self.direction_names}
            
            if cross_id in intersection_roads:
                road_count = sum(1 for k, v in intersection_roads[cross_id].items() 
                                if k in self.direction_names and v is True)
            else:
                road_count = 0
            intersection_progress.set_postfix({"Roads": f"{road_count}/8"})
        
        intersection_progress.close()
        
        return intersection_roads, (adjacency_list, edges)

    def load_all_csv_files(self):
        """Load and combine all CSV files from data folder"""
        print("📂 Loading CSV files from data folder...")
        
        csv_files = glob.glob(os.path.join(self.data_folder, "*.csv"))
        # Exclude SA_CROSS.csv from traffic data files
        csv_files = [f for f in csv_files if not os.path.basename(f).startswith('SA_CROSS')]
        
        print(f"🔍 Found {len(csv_files)} traffic data CSV files")
        
        if not csv_files:
            raise ValueError(f"❌ No CSV files found in {self.data_folder}")
        
        all_dataframes = []
        processed_files = 0
        
        # Progress bar for file loading
        file_progress = tqdm(csv_files, desc="📥 Loading files", unit="file", colour="blue")
        
        for file_path in file_progress:
            try:
                file_progress.set_postfix({"Current": os.path.basename(file_path)[:30]})
                
                df = pd.read_csv(file_path, encoding='utf-8')
                
                # Handle both 5MIN and 15MIN data
                if 'STAT_5MIN' in df.columns:
                    df['STAT_TIME'] = df['STAT_5MIN']
                    df['TIME_INTERVAL'] = '5min'
                elif 'STAT_15MIN' in df.columns:
                    df['STAT_TIME'] = df['STAT_15MIN']
                    df['TIME_INTERVAL'] = '15min'
                else:
                    continue
                
                # Basic data validation
                if 'CROSS_ID' not in df.columns or 'INFRA_TYPE' not in df.columns:
                    file_progress.write(f"⚠️  Skipping {os.path.basename(file_path)}: Missing required columns")
                    continue
                
                # Filter SMT data only
                smt_data = df[df['INFRA_TYPE'] == 'SMT'].copy()
                
                # Filter target intersections only
                smt_data = smt_data[smt_data['CROSS_ID'].isin(self.target_cross_ids)]
                
                if not smt_data.empty:
                    all_dataframes.append(smt_data)
                    processed_files += 1
                    
                file_progress.set_postfix({
                    "Processed": f"{processed_files}",
                    "Records": f"{len(smt_data) if not smt_data.empty else 0}"
                })
                    
            except Exception as e:
                file_progress.write(f"❌ Error processing {os.path.basename(file_path)}: {str(e)}")
                continue
        
        file_progress.close()
        
        if not all_dataframes:
            raise ValueError("❌ No valid data found in CSV files")
        
        # Combine all data with progress
        print("🔄 Combining all dataframes...")
        combined_data = pd.concat(all_dataframes, ignore_index=True)
        
        print(f"✅ Successfully loaded {processed_files} files")
        print(f"📊 Total records: {len(combined_data):,}")
        print("="*60)
        
        return combined_data

    def preprocess_temporal_data(self, data):
        """Preprocess data for temporal GNN input"""
        print("⏰ Preprocessing temporal data...")
        
        # Convert STAT_TIME to datetime
        print("📅 Converting timestamps...")
        tqdm.pandas(desc="Converting timestamps", colour="green")
        data['datetime'] = pd.to_datetime(data['STAT_TIME'].astype(str), format='%Y%m%d%H%M')
        
        # Sort by time and intersection
        print("🔀 Sorting data by time and intersection...")
        data = data.sort_values(['datetime', 'CROSS_ID']).reset_index(drop=True)
        
        # Remove duplicates
        print("🧹 Removing duplicates...")
        original_length = len(data)
        data = data.drop_duplicates(subset=['datetime', 'CROSS_ID'], keep='last')
        removed_duplicates = original_length - len(data)
        
        print(f"📅 Date range: {data['datetime'].min()} to {data['datetime'].max()}")
        print(f"⏰ Unique timestamps: {data['datetime'].nunique()}")
        print(f"🚦 Unique intersections: {data['CROSS_ID'].nunique()}")
        print(f"🧹 Removed duplicates: {removed_duplicates:,}")
        print("="*60)
        
        return data

    def create_node_features(self, data):
        """Create node feature matrix for each timestamp"""
        print("🎯 Creating node feature matrices...")
        
        # Get all unique timestamps
        timestamps = sorted(data['datetime'].unique())
        print(f"⏰ Processing {len(timestamps)} timestamps...")
        
        node_features_by_time = {}
        
        # Progress bar for timestamp processing
        timestamp_progress = tqdm(timestamps, desc="🎯 Processing timestamps", unit="timestamp", colour="cyan")
        
        for timestamp in timestamp_progress:
            timestamp_data = data[data['datetime'] == timestamp]
            
            # Create feature matrix for this timestamp
            features_matrix = []
            intersection_order = []
            
            for cross_id in self.target_cross_ids:
                cross_data = timestamp_data[timestamp_data['CROSS_ID'] == cross_id]
                
                if len(cross_data) > 0:
                    row = cross_data.iloc[0]
                    
                    # Extract features
                    features = []
                    
                    # Basic traffic features
                    features.append(row.get('VOL', 0))
                    features.append(row.get('VPHG', 0) if pd.notna(row.get('VPHG', 0)) else 0)
                    features.append(row.get('VS', 0) if pd.notna(row.get('VS', 0)) else 0)
                    
                    # Directional features (VOL_01 to VOL_24)
                    for i in range(1, 25):
                        vol_key = f'VOL_{i:02d}'
                        vol_val = row.get(vol_key, 0)
                        features.append(vol_val if pd.notna(vol_val) else 0)
                    
                    # Aggregate directional features (8 directions)
                    for direction, vol_indices in self.direction_mapping.items():
                        direction_volumes = []
                        for vol_idx in vol_indices:
                            vol_key = f'VOL_{vol_idx:02d}'
                            vol_val = row.get(vol_key, 0)
                            if pd.notna(vol_val):
                                direction_volumes.append(vol_val)
                        
                        # Average volume for this direction
                        avg_vol = np.mean(direction_volumes) if direction_volumes else 0
                        features.append(avg_vol)
                    
                    # Time-based features
                    hour = timestamp.hour
                    day_of_week = timestamp.weekday()
                    is_weekend = 1 if day_of_week >= 5 else 0
                    
                    features.extend([hour, day_of_week, is_weekend])
                    
                    # Add SA group features if available
                    if self.sa_cross_info is not None:
                        sa_info = self.sa_cross_info[self.sa_cross_info['CROSS_ID'] == cross_id]
                        if not sa_info.empty:
                            # One-hot encode district
                            district = sa_info['DIST'].iloc[0]
                            districts = ['A', 'B', 'C', 'D', 'E']  # Adjust based on actual districts
                            district_features = [1 if district == d else 0 for d in districts]
                            features.extend(district_features)
                        else:
                            features.extend([0] * 5)  # Default district features
                    
                    features_matrix.append(features)
                    intersection_order.append(cross_id)
                    
                else:
                    # Fill with zeros if no data
                    zero_features = [0] * (3 + 24 + 8 + 3 + (5 if self.sa_cross_info is not None else 0))
                    features_matrix.append(zero_features)
                    intersection_order.append(cross_id)
            
            node_features_by_time[timestamp] = {
                'features': np.array(features_matrix, dtype=np.float32),
                'intersection_order': intersection_order
            }
            
            # Update progress
            timestamp_progress.set_postfix({
                "Features": f"{len(features_matrix[0]) if features_matrix else 0}D",
                "Intersections": f"{len(intersection_order)}"
            })
        
        timestamp_progress.close()
        
        feature_dim = len(features_matrix[0]) if features_matrix else 0
        print(f"✅ Created node features for {len(timestamps)} timestamps")
        print(f"📊 Feature dimension per node: {feature_dim}")
        print("="*60)
        
        return node_features_by_time

    def create_temporal_sequences(self, node_features_by_time, sequence_length=12, prediction_horizon=3):
        """Create temporal sequences for training"""
        print(f"📈 Creating temporal sequences...")
        print(f"   🔄 Sequence length: {sequence_length}")
        print(f"   🎯 Prediction horizon: {prediction_horizon}")
        
        timestamps = sorted(node_features_by_time.keys())
        
        sequences = []
        targets = []
        
        max_sequences = len(timestamps) - sequence_length - prediction_horizon + 1
        
        # Progress bar
        sequence_progress = tqdm(range(max_sequences), 
                               desc="📈 Creating sequences", 
                               unit="sequence", colour="purple")
        
        for i in sequence_progress:
            # Input sequence
            input_sequence = []
            for j in range(sequence_length):
                timestamp = timestamps[i + j]
                features = node_features_by_time[timestamp]['features']
                input_sequence.append(features)
            
            # Target (future traffic volumes)
            target_timestamp = timestamps[i + sequence_length + prediction_horizon - 1]
            target_features = node_features_by_time[target_timestamp]['features']
            
            # Extract only directional traffic volumes as targets
            directional_targets = target_features[:, 3:27]  # VOL_01 to VOL_24
            
            sequences.append(np.array(input_sequence))
            targets.append(directional_targets)
            
            # Update progress
            sequence_progress.set_postfix({
                "Input Shape": f"{np.array(input_sequence).shape}",
                "Target Shape": f"{directional_targets.shape}"
            })
        
        sequence_progress.close()
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        print(f"✅ Created {len(sequences)} temporal sequences")
        print(f"📊 Input shape: {sequences.shape}")
        print(f"🎯 Target shape: {targets.shape}")
        print("="*60)
        
        return sequences, targets, timestamps

    def save_processed_data(self, node_features_by_time, adjacency_list, edges, sequences, targets, timestamps, intersection_roads):
        """Save all processed data for GNN training"""
        print("💾 Saving processed data...")
        
        save_tasks = [
            ("node_features.pkl", lambda: pickle.dump(node_features_by_time, open(os.path.join(self.output_folder, 'node_features.pkl'), 'wb'))),
            ("graph_structure.pkl", lambda: self._save_graph_structure(adjacency_list, edges, intersection_roads)),
            ("temporal_sequences.pkl", lambda: self._save_temporal_sequences(sequences, targets, timestamps)),
            ("metadata.json", lambda: self._save_metadata(sequences, targets, timestamps, edges)),
            ("CSV files", lambda: self.save_csv_outputs(edges, intersection_roads))
        ]
        
        # Progress bar for saving
        save_progress = tqdm(save_tasks, desc="💾 Saving files", unit="file", colour="green")
        
        for task_name, task_func in save_progress:
            save_progress.set_postfix({"Saving": task_name})
            task_func()
            time.sleep(0.1)
        
        save_progress.close()
        
        print(f"✅ All data saved to {self.output_folder}/")
        print("="*60)

    def _save_graph_structure(self, adjacency_list, edges, intersection_roads):
        """Save graph structure data"""
        graph_data = {
            'adjacency_list': adjacency_list,
            'edges': edges,
            'intersection_roads': intersection_roads,
            'target_cross_ids': self.target_cross_ids,
            'direction_mapping': self.direction_mapping,
            'uses_real_network': self.sa_cross_info is not None
        }
        
        graph_path = os.path.join(self.output_folder, 'graph_structure.pkl')
        with open(graph_path, 'wb') as f:
            pickle.dump(graph_data, f)

    def _save_temporal_sequences(self, sequences, targets, timestamps):
        """Save temporal sequence data"""
        sequences_data = {
            'sequences': sequences,
            'targets': targets,
            'timestamps': timestamps
        }
        
        sequences_path = os.path.join(self.output_folder, 'temporal_sequences.pkl')
        with open(sequences_path, 'wb') as f:
            pickle.dump(sequences_data, f)

    def _save_metadata(self, sequences, targets, timestamps, edges):
        """Save metadata"""
        metadata = {
            'num_intersections': len(self.target_cross_ids),
            'feature_dimension': sequences.shape[-1] if len(sequences) > 0 else 0,
            'num_directions': 24,
            'sequence_length': sequences.shape[1] if len(sequences) > 0 else 0,
            'num_sequences': len(sequences),
            'num_edges': len(edges),
            'uses_real_network': self.sa_cross_info is not None,
            'date_range': {
                'start': min(timestamps).isoformat() if timestamps else None,
                'end': max(timestamps).isoformat() if timestamps else None
            }
        }
        
        metadata_path = os.path.join(self.output_folder, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def save_csv_outputs(self, edges, intersection_roads):
        """Save CSV files for inspection"""
        # Save edge list with names
        edge_data = []
        for source, target in edges:
            source_info = self.sa_cross_info[self.sa_cross_info['CROSS_ID'] == source] if self.sa_cross_info is not None else None
            target_info = self.sa_cross_info[self.sa_cross_info['CROSS_ID'] == target] if self.sa_cross_info is not None else None
            
            edge_data.append({
                'source': source,
                'target': target,
                'source_name': source_info['CROSS_NAME'].iloc[0] if source_info is not None and not source_info.empty and 'CROSS_NAME' in source_info.columns else f'Cross_{source}',
                'target_name': target_info['CROSS_NAME'].iloc[0] if target_info is not None and not target_info.empty and 'CROSS_NAME' in target_info.columns else f'Cross_{target}'
            })
        
        edge_df = pd.DataFrame(edge_data)
        edge_path = os.path.join(self.output_folder, 'edge_list.csv')
        edge_df.to_csv(edge_path, index=False)
        
        # Save intersection roads with SA info
        roads_data = []
        for cross_id, roads_info in intersection_roads.items():
            road_info = {'intersection_id': cross_id}
            
            # Add road directions
            for direction in self.direction_names:
                road_info[direction] = roads_info.get(direction, False)
            
            # Add SA info if available (with underscore prefix)
            if '_sa_groups' in roads_info:
                road_info['sa_groups'] = ','.join(roads_info['_sa_groups'])
                road_info['district'] = roads_info.get('_district', '')
                road_info['name'] = roads_info.get('_name', f'Cross_{cross_id}')
            
            roads_data.append(road_info)
        
        roads_df = pd.DataFrame(roads_data)
        roads_path = os.path.join(self.output_folder, 'intersection_roads.csv')
        roads_df.to_csv(roads_path, index=False)

    def visualize_network(self):
        """Visualize the road network structure"""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            print("⚠️  matplotlib/networkx not installed. Skipping visualization.")
            return
        
        print("📊 Generating network visualization...")
        
        # Load saved graph structure
        with open(os.path.join(self.output_folder, 'graph_structure.pkl'), 'rb') as f:
            graph_data = pickle.load(f)
        
        edges = graph_data['edges']
        
        # Create NetworkX graph
        G = nx.Graph()
        G.add_edges_from(edges)
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes with different colors for different districts
        if self.sa_cross_info is not None:
            node_colors = []
            district_colors = {'A': 'red', 'B': 'blue', 'C': 'green', 'D': 'yellow', 'E': 'purple'}
            
            for node in G.nodes():
                node_info = self.sa_cross_info[self.sa_cross_info['CROSS_ID'] == node]
                if not node_info.empty:
                    district = node_info['DIST'].iloc[0]
                    node_colors.append(district_colors.get(district, 'gray'))
                else:
                    node_colors.append('gray')
        else:
            node_colors = 'lightblue'
        
        # Draw graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100)
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=6)
        
        plt.title("Traffic Intersection Network Graph")
        plt.axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'network_graph.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Network visualization saved to {self.output_folder}/network_graph.png")

    def run_full_preprocessing(self, sequence_length=12, prediction_horizon=3):
        """Run complete preprocessing pipeline"""
        print("🚀 " + "="*58)
        print("🚀 Starting GNN Data Preprocessing Pipeline")
        print("🚀 " + "="*58)
        start_time = time.time()

        # STEP 1: DB에서 15분 교통량 로딩 (전체 기간 배치)
        print("\n📂 STEP 1: Loading Database STAT_15MIN_CROSS")
        data = fetch_15min_from_db(
            start=None,                 # 전체 기간 자동 탐색
            end=None,
            cross_ids=self.target_cross_ids,   # 106개 대상만
            infra_type="SMT",
            batch_days=7               # 1주일 단위 배치 읽기 (필요시 3~14 조정)
        )
        
        # Step 2: Preprocess temporal data
        print("\n⏰ STEP 2: Preprocessing Temporal Data")
        data = self.preprocess_temporal_data(data)
        
        # Step 3: Create node features
        print("\n🎯 STEP 3: Creating Node Features")
        node_features_by_time = self.create_node_features(data)
        
        # Step 4: Analyze graph structure with real network
        print("\n🗺️  STEP 4: Analyzing Graph Structure")
        intersection_roads, (adjacency_list, edges) = self.analyze_graph_structure(data)
        
        # Step 5: Create temporal sequences
        print("\n📈 STEP 5: Creating Temporal Sequences")
        sequences, targets, timestamps = self.create_temporal_sequences(
            node_features_by_time, sequence_length, prediction_horizon
        )
        
        # Step 6: Save all processed data
        print("\n💾 STEP 6: Saving Processed Data")
        self.save_processed_data(
            node_features_by_time, adjacency_list, edges, 
            sequences, targets, timestamps, intersection_roads
        )
        
        # Step 7: Visualize network (optional)
        print("\n📊 STEP 7: Visualizing Network")
        self.visualize_network()
        
        # Final summary
        end_time = time.time()
        processing_time = end_time - start_time
        
        print("\n🎉 " + "="*58)
        print("🎉 GNN Data Preprocessing Complete!")
        print("🎉 " + "="*58)
        print(f"⏱️  Total processing time: {processing_time:.2f} seconds")
        print(f"📊 Dataset Summary:")
        print(f"   🚦 Intersections: {len(self.target_cross_ids)}")
        print(f"   ⏰ Time periods: {len(timestamps)}")
        print(f"   📈 Training sequences: {len(sequences):,}")
        print(f"   🔗 Graph edges: {len(edges)}")
        print(f"   🗺️  Network type: {'Real road network' if self.sa_cross_info is not None else 'Grid-based'}")
        print(f"   📊 Feature dimension: {sequences.shape[-1] if len(sequences) > 0 else 0}")
        print(f"   💾 Data size: {sequences.nbytes / (1024**2):.2f} MB")
        print("🎉 " + "="*58)
        
        return {
            'node_features': node_features_by_time,
            'adjacency_list': adjacency_list,
            'edges': edges,
            'sequences': sequences,
            'targets': targets,
            'timestamps': timestamps,
            'intersection_roads': intersection_roads,
            'processing_time': processing_time
        }

def main():
    """Main execution function"""
    print("🚦 Traffic GNN Data Preprocessor with Real Road Network")
    print("🚀 Starting preprocessing pipeline...")
    print()
    
    # Check if SA_CROSS.csv exists
    sa_cross_path = 'SA_CROSS.csv'
    if not os.path.exists(sa_cross_path):
        print(f"⚠️  {sa_cross_path} not found in current directory.")
        print("   Place SA_CROSS.csv in the current directory for real road network.")
        response = input("Continue with grid-based network? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Create preprocessor
    preprocessor = GNNTrafficDataPreprocessor(
        data_folder='data',           # Folder with raw CSV files
        output_folder='gnn_data',     # Output folder for processed data
        sa_cross_file=sa_cross_path   # SA_CROSS.csv path
    )
    
    # Run full preprocessing
    results = preprocessor.run_full_preprocessing(
        sequence_length=12,    # Use 12 time steps as input
        prediction_horizon=3   # Predict 3 time steps into future
    )
    
    return preprocessor, results

if __name__ == "__main__":
    try:
        preprocessor, results = main()
        print("\n✅ Preprocessing completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️  Preprocessing interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {str(e)}")
        raise