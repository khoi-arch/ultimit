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
    PATIENCE = 20
    LR = 2e-4
    WEIGHT_DECAY = 1e-3
    DATA_PATH = "Obfuscated-MalMem2022.csv" # Nhớ sửa lại nếu cần
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
        RAW_DF_CACHE.replace([np.inf, -np.inf], np.nan, inplace=True)
        RAW_DF_CACHE.dropna(inplace=True)
    return RAW_DF_CACHE.copy()

# ==========================================
# 1. DATA MODULE
# ==========================================
def get_data_old(file_path, seed):
    df = load_cached_df(file_path)
    y = df['Category'].apply(map_global_label).values
    X = df.drop(columns=['Class', 'Category'], errors='ignore').select_dtypes(include=[np.number]).fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_tmp, X_test, y_tmp, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.17647, random_state=seed, stratify=y_tmp) 
    
    return (torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_val, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32), 
            torch.tensor(y_train, dtype=torch.long), torch.tensor(y_val, dtype=torch.long), torch.tensor(y_test, dtype=torch.long), X.shape[1], 16)

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
# 2. ARCHITECTURES (CŨ, MỚI & HYBRID)
# ==========================================
# --- MODULES DÙNG CHUNG ---
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

class FeatureTokenizer_New(nn.Module):
    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_features, embed_dim))
        self.bias = nn.Parameter(torch.Tensor(num_features, embed_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)); nn.init.zeros_(self.bias)
    def forward(self, x):
        return x.unsqueeze(-1) * self.weight + self.bias

# --- MODEL 1: FULL BUNDLE CŨ (Chứa tham số use_noise) ---
class DifferentialAttentionBundle(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, d_model // num_heads 
        self.q1_h, self.k1_h = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.q2_h, self.k2_h = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.v_h, self.out_p = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.lambda_logits = nn.Parameter(torch.zeros(1, self.num_heads, 1, 1))
    def forward(self, x, use_noise=True):
        b, n, d = x.shape; sc = math.sqrt(self.head_dim)
        q1 = self.q1_h(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k1 = self.k1_h(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        q2 = self.q2_h(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k2 = self.k2_h(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v  = self.v_h(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        l1, l2 = torch.matmul(q1, k1.transpose(-2, -1))/(sc+1e-6), torch.matmul(q2, k2.transpose(-2, -1))/(sc+1e-6)
        
        if self.training and use_noise: l2 += 0.1 * torch.randn_like(l2)
            
        a1, a2 = torch.softmax(l1, -1), torch.softmax(l2, -1)
        lam = torch.sigmoid(self.lambda_logits)
        out = torch.matmul((a1 - lam * a2)/(1.0 + lam), v).transpose(1, 2).contiguous().view(b, n, d)
        return self.out_p(out), a1, a2, lam

class DiffTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = DifferentialAttentionBundle(d_model, num_heads)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model))
    def forward(self, x, use_noise=True):
        out, a1, a2, lam = self.attn(self.norm1(x), use_noise=use_noise)
        x = x + out
        return x + self.ffn(self.norm2(x)), a1, a2, lam

class OldModelBundle(nn.Module):
    def __init__(self, num_features, d_model=128, num_heads=4, num_classes=16, num_layers=3, use_noise=True):
        super().__init__()
        self.tokenizer = FTTransformerTokenizer_Old(num_features, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.blocks = nn.ModuleList([DiffTransformerBlock(d_model, num_heads) for _ in range(num_layers)])
        self.shared = nn.Sequential(nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(0.2))
        self.h2, self.h4, self.h16 = nn.Linear(256, 2), nn.Linear(256, 4), nn.Linear(256, num_classes)
        self.log_vars = nn.Parameter(torch.zeros(3))
        self.use_noise_flag = use_noise  # FIX: Có thể bật tắt

    def forward(self, x):
        x = torch.clamp(x, min=-5.0, max=5.0) 
        b = x.shape[0]; tokens = self.tokenizer(x); cls = self.cls_token.expand(b, -1, -1)
        attn_x = torch.cat((cls, tokens), dim=1); a1l, a2l, lams = [], [], []
        
        for block in self.blocks: 
            attn_x, a1, a2, lam = block(attn_x, use_noise=self.use_noise_flag)
            a1l.append(a1); a2l.append(a2); lams.append(lam)
        
        rep = self.shared(attn_x[:, 0, :])
        logits = (self.h2(rep), self.h4(rep), self.h16(rep))
        
        if self.training: return {"logits": logits, "attn": (a1l, a2l), "lambdas": lams}
        return {"logits": logits}

# --- MODEL 2: NEW MODEL (Baseline cho Test 2) ---
class NewModel(nn.Module):
    def __init__(self, num_features, num_classes=16, embed_dim=128):
        super().__init__()
        self.pre_norm = nn.LayerNorm(num_features)
        self.tokenizer = FeatureTokenizer_New(num_features, embed_dim)
        self.token_norm = nn.LayerNorm(embed_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=embed_dim*4, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        x = self.pre_norm(x)
        x = self.token_norm(self.tokenizer(x))
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)
        return self.head(x[:, 0])

# --- MODEL 3: HYBRID MODEL (Giải pháp giải quyết Test 2 của bạn) ---
class HybridNewModel(nn.Module):
    def __init__(self, num_features, num_classes=16, embed_dim=128):
        super().__init__()
        # Tokenizer Sin/Cos Cũ
        self.tokenizer = FTTransformerTokenizer_Old(num_features, embed_dim)
        # CẦU NỐI KỲ DIỆU TẠI ĐÂY (Thuần hóa distribution cho PyTorch Transformer)
        self.bridge_norm = nn.LayerNorm(embed_dim)
        
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=embed_dim*4, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        x = torch.clamp(x, min=-5.0, max=5.0) # Bắt buộc clamp cho Sin/Cos
        tokens = self.tokenizer(x)
        tokens = self.bridge_norm(tokens) # Cân bằng Distribution
        
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        x = self.transformer(x)
        return self.head(x[:, 0])


# ==========================================
# 3. TRAINING ENGINES 
# ==========================================
class FocalLossWithSmoothing(nn.Module):
    def __init__(self, weight=None, gamma=1.5, label_smoothing=0.05):
        super().__init__()
        self.weight, self.gamma, self.eps = weight, gamma, label_smoothing
    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.eps)
        loss = ((1 - torch.exp(-ce)) ** self.gamma) * ce
        if self.weight is not None: loss *= self.weight[targets].float()
        return loss.mean()

def create_mapping_matrix():
    M = torch.zeros(16, 4, device=Config.DEVICE)
    M[0, 0] = 1.0; M[[1, 2, 3, 4, 5], 1] = 1.0; M[[6, 7, 8, 9, 10], 2] = 1.0; M[[11, 12, 13, 14, 15], 3] = 1.0             
    return M

def generate_hierarchical_labels(y_16):
    dev = y_16.device
    y_2 = (y_16 > 0).long()
    y_4 = torch.zeros_like(y_16)
    y_4[torch.isin(y_16, torch.tensor([1, 2, 3, 4, 5], device=dev))] = 1
    y_4[torch.isin(y_16, torch.tensor([6, 7, 8, 9, 10], device=dev))] = 2
    y_4[torch.isin(y_16, torch.tensor([11, 12, 13, 14, 15], device=dev))] = 3
    return y_2, y_4

def evaluate_loader(model, loader):
    model.eval()
    preds_list, labels_list = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(Config.DEVICE))
            if isinstance(logits, dict): logits = logits["logits"][2]
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds_list.extend(preds)
            labels_list.extend(y.numpy())
    return f1_score(labels_list, preds_list, average='macro', zero_division=0)

def run_unified_training(model, train_loader, val_loader, test_loader, engine_type='new'):
    model = model.to(Config.DEVICE)
    best_val_f1 = 0.0; best_state = None; patience_counter = 0
    
    all_y = train_loader.dataset.tensors[1].cpu().numpy()
    counts = np.bincount(all_y, minlength=16)
    weights = len(all_y) / (16 * np.where(counts == 0, 1, counts))
    class_weights = torch.tensor(weights, dtype=torch.float32).to(Config.DEVICE)
    
    if engine_type == 'new':
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
    elif engine_type == 'old_bundle': 
        criterion_focal = FocalLossWithSmoothing(weight=class_weights)
        M_16_to_4 = create_mapping_matrix()
        optimizer = optim.AdamW([
            {'params': [p for n, p in model.named_parameters() if n != 'log_vars'], 'lr': 1e-3},
            {'params': [model.log_vars], 'lr': 5e-3}
        ], weight_decay=1e-4)

    for epoch in range(Config.EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            optimizer.zero_grad()
            
            if engine_type == 'old_bundle':
                by2, by4 = generate_hierarchical_labels(y)
                output = model(x)
                l2, l4, l16 = output["logits"]
                loss_16 = criterion_focal(l16, y); loss_2 = F.cross_entropy(l2, by2); loss_4 = F.cross_entropy(l4, by4)
                
                pre = torch.exp(-model.log_vars)
                loss = (0.5*pre[0]*loss_2+model.log_vars[0]) + (0.5*pre[1]*loss_4+model.log_vars[1]) + (0.5*pre[2]*loss_16+model.log_vars[2]) + 0.001*(model.log_vars**2).sum()
                
                p4h = torch.matmul(F.softmax(l16, dim=1), M_16_to_4)
                lcon = F.nll_loss(torch.log(p4h + 1e-8), by4)
                loss += 0.3 * lcon 
                loss -= 0.1 * (-(p4h * torch.log(p4h + 1e-8)).sum(1) / math.log(4)).mean()
                
                for lam in output["lambdas"]: loss += 0.01 * ((lam - 0.5) ** 2).mean() 
                for a1, a2 in zip(output["attn"][0], output["attn"][1]):
                    loss += 0.05 * (torch.matmul(F.normalize(a1.flatten(2), dim=-1), F.normalize(a2.flatten(2), dim=-1).transpose(-2, -1)) ** 2).mean()
            else:
                logits = model(x)
                loss = criterion(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_f1 = evaluate_loader(model, val_loader)
        if engine_type == 'new': scheduler.step(val_f1)

        if val_f1 > best_val_f1: 
            best_val_f1 = val_f1; best_state = {k: v.cpu() for k, v in model.state_dict().items()}; patience_counter = 0
        else: patience_counter += 1
            
        if patience_counter >= Config.PATIENCE: break

    model.load_state_dict(best_state)
    return evaluate_loader(model, test_loader)

# ==========================================
# 4. EXPERIMENT CONTROLLER (PHASE 2 - ABLATION)
# ==========================================
def run_experiments():
    try: load_cached_df(Config.DATA_PATH)
    except FileNotFoundError: return

    # --- KẾ HOẠCH PHASE 2 CHUẨN XÁC CỦA BẠN ---
    configs = [
        # TEST 1: Noise Effect
        {"name": "T1: Old Data + Old Bundle (WITH Noise)", "data": "old", "model": "old_bundle", "engine": "old_bundle", "noise": True},
        {"name": "T1: Old Data + Old Bundle (NO Noise)", "data": "old", "model": "old_bundle", "engine": "old_bundle", "noise": False},
        
        # TEST 2: Tokenizer Compatibility Effect
        {"name": "T2: New Data + New Model (Linear Tokenizer)", "data": "new", "model": "new_standard", "engine": "new", "noise": False},
        {"name": "T2: New Data + Hybrid Model (SinCos + LayerNorm + Transf)", "data": "new", "model": "hybrid", "engine": "new", "noise": False},
        
        # TEST 3: Holy Grail
        {"name": "T3 (Holy Grail): New Data + Old Bundle (WITH Noise)", "data": "new", "model": "old_bundle", "engine": "old_bundle", "noise": True},
        {"name": "T3 (Holy Grail): New Data + Old Bundle (NO Noise)", "data": "new", "model": "old_bundle", "engine": "old_bundle", "noise": False},
    ]

    all_results = {cfg["name"]: [] for cfg in configs}
    print("\n🚀 BẮT ĐẦU PHASE 2: CONTROLLED ABLATION (THE FINAL JUDGMENT)...")
    print("-" * 80)

    for seed in Config.SEEDS:
        print(f"\n🌱 ĐANG CHẠY SEED: {seed}")
        seed_everything(seed)
        d_old = get_data_old(Config.DATA_PATH, seed); d_new = get_data_new(Config.DATA_PATH, seed)

        for cfg in configs:
            print(f"🔄 Test: {cfg['name']}")
            X_tr, X_val, X_te, y_tr, y_val, y_te, num_feat, num_class = d_old if cfg['data'] == 'old' else d_new
            train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=Config.BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=Config.BATCH_SIZE)
            test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=Config.BATCH_SIZE)

            if cfg['model'] == 'old_bundle': model = OldModelBundle(num_features=num_feat, num_classes=num_class, use_noise=cfg['noise'])
            elif cfg['model'] == 'hybrid': model = HybridNewModel(num_features=num_feat, num_classes=num_class)
            else: model = NewModel(num_features=num_feat, num_classes=num_class)

            test_f1 = run_unified_training(model, train_loader, val_loader, test_loader, engine_type=cfg['engine'])
            all_results[cfg['name']].append(test_f1)
            
            del model; torch.cuda.empty_cache(); gc.collect()

    print("\n🏆 BẢNG KẾT QUẢ TỔNG HỢP PHASE 2:")
    print("=" * 100)
    final_report = []
    for name, scores in all_results.items():
        mean_f1 = np.mean(scores); std_f1 = np.std(scores)
        final_report.append({"Experiment": name, "Mean_F1_Test": round(mean_f1, 4), "Std": round(std_f1, 4)})
        print(f"{name:<70} | F1: {mean_f1:.4f} ± {std_f1:.4f}")
    pd.DataFrame(final_report).to_csv("Phase2_Ablation_Results.csv", index=False)
    print("=" * 100)

if __name__ == "__main__":
    run_experiments()