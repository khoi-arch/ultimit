import os
import gc
import time
import math
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=UserWarning) # Tắt warning cho sạch màn hình

# ==========================================
# 0. CONFIG & SETUP (PHASE 3)
# ==========================================
class Config:
    # CHẠY 3 SEED ĐỂ ĐO ĐỘ ỔN ĐỊNH CỰC KỲ CHÍNH XÁC
    SEEDS = [42, 2024, 9999]  
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 128
    EPOCHS = 100
    PATIENCE = 15 # Ép hội tụ nhanh hơn, không lề mề
    DATA_PATH = "Obfuscated-MalMem2022.csv" # Kiểm tra lại path nếu cần

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

GLOBAL_CLASSES = [
    'benign', 'zeus', 'emotet', 'refroso', 'scar', 'reconyc',
    '180solutions', 'coolwebsearch', 'gator', 'transponder',
    'tibs', 'conti', 'maze', 'pysa', 'ako', 'shade'
]
CLASS_TO_IDX = {k: i for i, k in enumerate(GLOBAL_CLASSES)}

def map_global_label(c):
    c = str(c).lower()
    if 'benign' in c: return 0
    if 'cws' in c: return CLASS_TO_IDX['coolwebsearch']
    for k, v in CLASS_TO_IDX.items():
        if k in c: return v
    return 0

RAW_DF_CACHE = None
def load_cached_df(file_path):
    global RAW_DF_CACHE
    if RAW_DF_CACHE is None:
        RAW_DF_CACHE = pd.read_csv(file_path)
        RAW_DF_CACHE.columns = RAW_DF_CACHE.columns.str.strip()
        RAW_DF_CACHE.replace([np.inf, -np.inf], np.nan, inplace=True)
        RAW_DF_CACHE.dropna(inplace=True)
    return RAW_DF_CACHE.copy()

# ==========================================
# 1. DATA MODULE (CHỈ DÙNG NEW DATA)
# ==========================================
def get_data_new(file_path, seed):
    df = load_cached_df(file_path)
    y = df['Category'].apply(map_global_label).values
    X = df.drop(columns=['Class', 'Category'], errors='ignore').select_dtypes(include=[np.number]).fillna(0)
    
    Q1 = X.quantile(0.25); Q3 = X.quantile(0.75); IQR = Q3 - Q1
    X = X.clip(lower=Q1 - 3*IQR, upper=Q3 + 3*IQR, axis=1)
    X = np.sign(X) * np.sqrt(np.abs(X))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_tmp, X_test, y_tmp, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.17647, random_state=seed, stratify=y_tmp)
    
    return (torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_val, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32), 
            torch.tensor(y_train, dtype=torch.long), torch.tensor(y_val, dtype=torch.long), torch.tensor(y_test, dtype=torch.long), X.shape[1], 16)

# ==========================================
# 2. ARCHITECTURE: HYBRID ULTIMATE
# ==========================================
class FTTransformerTokenizer_Old(nn.Module):
    def __init__(self, num_features, d_model, k=4):
        super().__init__()
        self.frequencies = nn.Parameter(torch.randn(num_features, k))
        self.mlp = nn.Linear(k * 2, d_model)
        self.column_embeddings = nn.Parameter(torch.randn(1, num_features, d_model) * 0.02)
    def forward(self, x):
        angles = 2 * math.pi * x.unsqueeze(-1) * self.frequencies
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1) 
        return self.mlp(emb) + self.column_embeddings.expand(x.size(0), -1, -1)

class HybridNewModel(nn.Module):
    # FIX: Mở khóa tham số dropout_rate để tinh chỉnh
    def __init__(self, num_features, num_classes=16, embed_dim=128, dropout_rate=0.0):
        super().__init__()
        self.tokenizer = FTTransformerTokenizer_Old(num_features, embed_dim)
        self.bridge_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate) # Lớp khiên chống học vẹt
        
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=embed_dim*4, 
                                                   batch_first=True, norm_first=True, dropout=dropout_rate)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        x = torch.clamp(x, min=-5.0, max=5.0) 
        tokens = self.tokenizer(x)
        tokens = self.bridge_norm(tokens)
        tokens = self.dropout(tokens) # Ép tính ổn định
        
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        x = self.transformer(x)
        return self.head(x[:, 0])

# ==========================================
# 3. TRAINING ENGINE (TỐI ƯU HÓA)
# ==========================================
def evaluate_loader(model, loader):
    model.eval()
    preds_list, labels_list = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(Config.DEVICE))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds_list.extend(preds)
            labels_list.extend(y.numpy())
    return f1_score(labels_list, preds_list, average='macro', zero_division=0)

def run_hybrid_training(model, train_loader, val_loader, test_loader, lr=2e-4, wd=1e-3):
    model = model.to(Config.DEVICE)
    best_val_f1 = 0.0; best_state = None; patience_counter = 0
    
    all_y = train_loader.dataset.tensors[1].cpu().numpy()
    counts = np.bincount(all_y, minlength=16)
    weights = len(all_y) / (16 * np.where(counts == 0, 1, counts))
    class_weights = torch.tensor(weights, dtype=torch.float32).to(Config.DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)

    for epoch in range(Config.EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_f1 = evaluate_loader(model, val_loader)
        scheduler.step(val_f1)

        if val_f1 > best_val_f1: 
            best_val_f1 = val_f1; best_state = {k: v.cpu() for k, v in model.state_dict().items()}; patience_counter = 0
        else: patience_counter += 1
            
        if patience_counter >= Config.PATIENCE: break

    model.load_state_dict(best_state)
    return evaluate_loader(model, test_loader)

# ==========================================
# 4. EXPERIMENT CONTROLLER (PHASE 3: STABILIZATION)
# ==========================================
def run_experiments():
    try: load_cached_df(Config.DATA_PATH)
    except FileNotFoundError: 
        print(f"❌ LỖI: Không tìm thấy file data tại {Config.DATA_PATH}! Hãy check lại đường dẫn.")
        return

    # --- KẾ HOẠCH PHASE 3: KÌM CƯƠNG VỊ VUA ---
    configs = [
        # Baseline của Vua (Lấy từ Phase 2)
        {"name": "P3_Base: Hybrid + LR 2e-4 + No Dropout", "lr": 2e-4, "wd": 1e-3, "dropout": 0.0},
        
        # Test 1: Giảm LR để mô hình hội tụ "mượt" hơn, tránh vấp váp
        {"name": "P3_T1: Hybrid + Low LR (5e-5)", "lr": 5e-5, "wd": 1e-3, "dropout": 0.0},
        
        # Test 2: Bật Dropout để chống Overfit, ép std giảm
        {"name": "P3_T2: Hybrid + Dropout (0.2)", "lr": 2e-4, "wd": 1e-3, "dropout": 0.2},
        
        # Test 3: Ép trọng số nhỏ lại (Tăng Weight Decay)
        {"name": "P3_T3: Hybrid + High Weight Decay (1e-2)", "lr": 2e-4, "wd": 1e-2, "dropout": 0.0},
        
        # Test 4: Combo tổng hợp (Vừa đủ)
        {"name": "P3_T4_Ultimate: Low LR + Dropout 0.1 + High WD", "lr": 1e-4, "wd": 1e-2, "dropout": 0.1},
    ]

    all_results = {cfg["name"]: [] for cfg in configs}
    print("\n🚀 BẮT ĐẦU PHASE 3: THE FINAL TUNE (ÉP STD XUỐNG ĐÁY)...")
    print("-" * 80)

    for seed in Config.SEEDS:
        print(f"\n🌱 ĐANG CHẠY SEED: {seed}")
        seed_everything(seed)
        d_new = get_data_new(Config.DATA_PATH, seed)

        for cfg in configs:
            print(f"🔄 Test: {cfg['name']}")
            X_tr, X_val, X_te, y_tr, y_val, y_te, num_feat, num_class = d_new
            train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=Config.BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=Config.BATCH_SIZE)
            test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=Config.BATCH_SIZE)

            model = HybridNewModel(num_features=num_feat, num_classes=num_class, dropout_rate=cfg['dropout'])

            test_f1 = run_hybrid_training(model, train_loader, val_loader, test_loader, lr=cfg['lr'], wd=cfg['wd'])
            all_results[cfg['name']].append(test_f1)
            
            del model; torch.cuda.empty_cache(); gc.collect()

    print("\n🏆 BẢNG KẾT QUẢ TỔNG HỢP PHASE 3:")
    print("=" * 100)
    final_report = []
    for name, scores in all_results.items():
        mean_f1 = np.mean(scores); std_f1 = np.std(scores)
        final_report.append({"Experiment": name, "Mean_F1_Test": round(mean_f1, 4), "Std": round(std_f1, 4)})
        print(f"{name:<60} | F1: {mean_f1:.4f} ± {std_f1:.4f}")
    pd.DataFrame(final_report).to_csv("Phase3_FinalTune_Results.csv", index=False)
    print("=" * 100)

if __name__ == "__main__":
    run_experiments()