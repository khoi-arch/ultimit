import os
import gc
import time
import json
import math
import torch
import joblib
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ==========================================
# 0. CONFIG & SETUP (PHASE 0)
# ==========================================
class Config:
    SEEDS = [42, 2024]  
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 128
    EPOCHS = 100
    LR = 2e-4
    WEIGHT_DECAY = 1e-3
    DATA_PATH = "Obfuscated-MalMem2022.csv"
    PROCESSED_DIR = "./processed_data"

os.makedirs(Config.PROCESSED_DIR, exist_ok=True)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# GLOBAL MAPPING 
GLOBAL_CLASSES = [
    'benign', 'zeus', 'emotet', 'refroso', 'scar', 'reconyc',
    '180solutions', 'coolwebsearch', 'gator', 'transponder',
    'tibs', 'conti', 'maze', 'pysa', 'ako', 'shade'
]
CLASS_TO_IDX = {k: i for i, k in enumerate(GLOBAL_CLASSES)}

def map_global_label(c):
    c = str(c).lower()
    if 'benign' in c: return 0
    # FIX: Bắt cứng từ khóa viết tắt để không bị rớt class
    if 'cws' in c: return CLASS_TO_IDX['coolwebsearch']
    for k, v in CLASS_TO_IDX.items():
        if k in c: return v
    return 0

RAW_DF_CACHE = None

def load_cached_df(file_path):
    global RAW_DF_CACHE
    if RAW_DF_CACHE is None:
        print(f"[*] Đọc CSV từ đĩa: {file_path}")
        RAW_DF_CACHE = pd.read_csv(file_path)
        RAW_DF_CACHE.columns = RAW_DF_CACHE.columns.str.strip()
        # FIX: Phải dọn inf trước khi dropna
        RAW_DF_CACHE.replace([np.inf, -np.inf], np.nan, inplace=True)
        RAW_DF_CACHE.dropna(inplace=True)
    return RAW_DF_CACHE.copy()

# ==========================================
# 1. DATA MODULE (BẢN CHUẨN HÓA CHO AUTO-TEST)
# ==========================================
def get_data_old(file_path, seed):
    df = load_cached_df(file_path)
    y = df['Category'].apply(map_global_label).values
    X = df.drop(columns=['Class', 'Category'], errors='ignore')
    
    X = X.select_dtypes(include=[np.number]).fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_tmp, X_test, y_tmp, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.17647, random_state=seed, stratify=y_tmp) 
    
    return (torch.tensor(X_train, dtype=torch.float32), 
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(X_test, dtype=torch.float32), 
            torch.tensor(y_train, dtype=torch.long), 
            torch.tensor(y_val, dtype=torch.long),
            torch.tensor(y_test, dtype=torch.long), 
            X.shape[1], 16)

def get_data_new(file_path, seed):
    df = load_cached_df(file_path)
    y = df['Category'].apply(map_global_label).values
    X = df.drop(columns=['Class', 'Category'], errors='ignore').select_dtypes(include=[np.number]).fillna(0)
    
    # 1. Clipping (Outlier)
    Q1 = X.quantile(0.25); Q3 = X.quantile(0.75); IQR = Q3 - Q1
    X = X.clip(lower=Q1 - 3*IQR, upper=Q3 + 3*IQR, axis=1)
    
    # 2. FIX: Phục hồi Hybrid Scaler (Signed SQRT) để trị phân phối lệch
    X = np.sign(X) * np.sqrt(np.abs(X))
    
    # 3. Standard Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_tmp, X_test, y_tmp, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.17647, random_state=seed, stratify=y_tmp)
    
    return (torch.tensor(X_train, dtype=torch.float32), 
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(X_test, dtype=torch.float32), 
            torch.tensor(y_train, dtype=torch.long), 
            torch.tensor(y_val, dtype=torch.long),
            torch.tensor(y_test, dtype=torch.long), 
            X.shape[1], 16)

# ==========================================
# 2. ARCHITECTURES (CŨ VÀ MỚI)
# ==========================================
# --- BẢN CŨ ---
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

class DifferentialAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, d_model // num_heads 
        self.q1_h = nn.Linear(d_model, d_model); self.k1_h = nn.Linear(d_model, d_model)
        self.q2_h = nn.Linear(d_model, d_model); self.k2_h = nn.Linear(d_model, d_model)
        self.v_h = nn.Linear(d_model, d_model); self.out_p = nn.Linear(d_model, d_model)
        self.lambda_logits = nn.Parameter(torch.zeros(1, self.num_heads, 1, 1))
    def forward(self, x, use_noise=True):
        b, n, d = x.shape; sc = math.sqrt(self.head_dim)
        q1 = self.q1_h(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k1 = self.k1_h(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        q2 = self.q2_h(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k2 = self.k2_h(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v  = self.v_h(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        l1 = torch.matmul(q1, k1.transpose(-2, -1))/(sc+1e-6)
        l2 = torch.matmul(q2, k2.transpose(-2, -1))/(sc+1e-6)
        
        if self.training and use_noise: 
            l2 += 0.1 * torch.randn_like(l2)
            
        a1, a2 = torch.softmax(l1, -1), torch.softmax(l2, -1)
        lam = torch.sigmoid(self.lambda_logits)
        out = torch.matmul((a1 - lam * a2)/(1.0 + lam), v).transpose(1, 2).contiguous().view(b, n, d)
        return self.out_p(out)

class OldModel(nn.Module):
    def __init__(self, num_features, num_classes=16, d_model=128, use_noise=True):
        super().__init__()
        self.tokenizer = FTTransformerTokenizer_Old(num_features, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = DifferentialAttention(d_model, 4)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model))
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.use_noise_flag = use_noise 
        
    def forward(self, x):
        x = torch.clamp(x, min=-5.0, max=5.0) 
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, tokens), dim=1)
        x = x + self.attn(self.norm1(x), self.use_noise_flag)
        x = x + self.ffn(self.norm2(x))
        return self.head(x[:, 0, :])

# --- BẢN MỚI ---
class FeatureTokenizer_New(nn.Module):
    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_features, embed_dim))
        self.bias = nn.Parameter(torch.Tensor(num_features, embed_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)); nn.init.zeros_(self.bias)
    def forward(self, x):
        return x.unsqueeze(-1) * self.weight + self.bias

class NewModel(nn.Module):
    def __init__(self, num_features, num_classes=16, embed_dim=128):
        super().__init__()
        self.pre_norm = nn.LayerNorm(num_features)
        self.tokenizer = FeatureTokenizer_New(num_features, embed_dim)
        self.token_norm = nn.LayerNorm(embed_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=embed_dim*4, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        x = self.pre_norm(x)
        x = self.token_norm(self.tokenizer(x))
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)
        return self.head(x[:, 0])

# ==========================================
# 3. TRAINING ENGINES (UNIFIED ADAPTER)
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

def run_unified_training(model, train_loader, val_loader, test_loader, engine_type='new'):
    model = model.to(Config.DEVICE)
    best_val_f1 = 0.0
    best_state = None
    
    # FIX: Tính Weights bằng Numpy thuần dựa trên tensors gốc, miễn nhiễm 100% với lỗi 0-dim của Dataloader
    all_y = train_loader.dataset.tensors[1].cpu().numpy()
    num_classes = 16
    counts = np.bincount(all_y, minlength=num_classes)
    counts_safe = np.where(counts == 0, 1, counts)
    weights = len(all_y) / (num_classes * counts_safe)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(Config.DEVICE)
    
    if engine_type == 'new':
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    else:
        def focal_loss(inputs, targets):
            ce = F.cross_entropy(inputs, targets, reduction='none')
            return (((1 - torch.exp(-ce)) ** 1.5) * ce).mean()
        criterion = focal_loss
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(Config.EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_f1 = evaluate_loader(model, val_loader)
        if val_f1 > best_val_f1: 
            best_val_f1 = val_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
    model.load_state_dict(best_state)
    test_f1 = evaluate_loader(model, test_loader)
    
    return test_f1

# ==========================================
# 4. EXPERIMENT CONTROLLER (AUTOMATION LOOP)
# ==========================================
def run_experiments():
    try:
        load_cached_df(Config.DATA_PATH)
    except FileNotFoundError:
        print(f"❌ LỖI: Không tìm thấy file data tại {Config.DATA_PATH}!")
        return

    configs = [
        # --- MODE A: FAIR COMPARISON ---
        {"name": "C1 (Fair): Old Data + Old Model + New Engine (No Noise)", "data": "old", "model": "old", "engine": "new", "noise": False},
        {"name": "C2 (Fair): New Data + New Model + New Engine", "data": "new", "model": "new", "engine": "new", "noise": False},
        {"name": "C3 (Fair): New Data + Old Model + New Engine (No Noise)", "data": "new", "model": "old", "engine": "new", "noise": False},
        {"name": "C4 (Fair): Old Data + New Model + New Engine", "data": "old", "model": "new", "engine": "new", "noise": False},
        
        # --- MODE B: FULL BUNDLE ---
        {"name": "C1 (Bundle): Old Data + Old Model + Old Engine (With Noise)", "data": "old", "model": "old", "engine": "old", "noise": True},
    ]

    all_results = {cfg["name"]: [] for cfg in configs}

    print("\n🚀 BẮT ĐẦU THỰC NGHIỆM CHÍNH XÁC CAO (MULTI-SEED + NO LEAKAGE)...")
    print("-" * 70)

    for seed in Config.SEEDS:
        print(f"\n🌱 ĐANG CHẠY SEED: {seed}")
        seed_everything(seed)
        
        d_old = get_data_old(Config.DATA_PATH, seed)
        d_new = get_data_new(Config.DATA_PATH, seed)

        for cfg in configs:
            print(f"🔄 Test: {cfg['name']}")
            
            X_tr, X_val, X_te, y_tr, y_val, y_te, num_feat, num_class = d_old if cfg['data'] == 'old' else d_new
            
            train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=Config.BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=Config.BATCH_SIZE)
            test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=Config.BATCH_SIZE)

            if cfg['model'] == 'old':
                model = OldModel(num_features=num_feat, num_classes=num_class, use_noise=cfg['noise'])
            else:
                model = NewModel(num_features=num_feat, num_classes=num_class)

            test_f1 = run_unified_training(model, train_loader, val_loader, test_loader, engine_type=cfg['engine'])
            all_results[cfg['name']].append(test_f1)
            
            del model
            torch.cuda.empty_cache()
            gc.collect()

    print("\n🏆 BẢNG KẾT QUẢ TỔNG HỢP PHASE 1 (AVERAGED OVER SEEDS):")
    print("=" * 80)
    
    final_report = []
    for name, scores in all_results.items():
        mean_f1 = np.mean(scores)
        std_f1 = np.std(scores)
        final_report.append({"Experiment": name, "Mean_F1_Test": round(mean_f1, 4), "Std": round(std_f1, 4)})
        print(f"{name:<60} | F1: {mean_f1:.4f} ± {std_f1:.4f}")
        
    df_res = pd.DataFrame(final_report)
    df_res.to_csv("Phase1_Results_Robust.csv", index=False)
    print("=" * 80)

if __name__ == "__main__":
    run_experiments()