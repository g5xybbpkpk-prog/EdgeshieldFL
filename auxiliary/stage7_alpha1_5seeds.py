# 1. Environment Setup & Hardware Verification
import sys
import os
import time
import math
import copy
import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=" * 70)
print(f"PyTorch Version: {torch.__version__}")
print(f"Execution Device: {device}")
if torch.cuda.is_available():
    print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Available VRAM:  {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: CUDA not detected! Running on CPU will be significantly slower.")
    print("Please enable GPU accelerator before running.")
print("=" * 70)



# ======================================================================


# 2. Architecture & EdgeShield-FL Mathematical Constants

# Model Architecture: Exact 4-layer CNN (d = 136,874 parameters)
class EdgeCNN(nn.Module):
    def __init__(self):
        super(EdgeCNN, self).__init__()
        # Conv1: 3 in, 16 out, 3x3, pad=1 -> 16*3*9 + 16 = 448
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        # Conv2: 16 in, 32 out, 3x3, pad=1 -> 32*16*9 + 32 = 4,640
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # 32x32 -> Pool -> 16x16 -> Pool -> 8x8. 32 channels * 8 * 8 = 2048
        # FC1: 2048 in, 64 out -> 64*2048 + 64 = 131,136
        self.fc1 = nn.Linear(2048, 64)
        # FC2: 64 in, 10 out -> 10*64 + 10 = 650
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_layer_slices(self):
        """Returns parameter slice boundaries for the 4 layers: Conv1, Conv2, FC1, FC2."""
        slices = []
        curr = 0
        for p in [list(self.conv1.parameters()), list(self.conv2.parameters()), 
                  list(self.fc1.parameters()), list(self.fc2.parameters())]:
            p_len = sum(param.numel() for param in p)
            slices.append((curr, curr + p_len))
            curr += p_len
        return slices

# Verify exact parameter count
sample_model = EdgeCNN()
d_params = sum(p.numel() for p in sample_model.parameters())
print(f"EdgeCNN Instantiated. Total Parameters d = {d_params:,} (Target: 136,874)")
assert d_params == 136874, f"Parameter count mismatch: {d_params} != 136874"

# Global Federated Hyperparameters
NUM_CLIENTS = 20             # N = 20 edge devices
LOCAL_EPOCHS = 2             # E = 2 local epochs per round
BATCH_SIZE = 64              # B = 64
LR_LOCAL = 0.08              # Local SGD learning rate
CLIP_C0 = 0.5                # Per-sample clip bound
CLIP_C = 5.0                 # Update-level L2 clip bound C (Frozen from Stage 2 sweep)
BATCH_CLIP_NORM = 5.0        # Per-batch gradient clip bound

# EdgeShield-FL Defense Parameters
M_SKETCH = 64                # Rademacher sketch dimension m = 64
SIGMA_S = 0.05               # Sketch perturbation std
SIGMA_PROF = 0.01            # Profile perturbation std
B_M_LAPLACE = 0.05           # Laplace norm noise scale
LAMBDA_MAD = 1.8             # MAD multiplier
LAMBDA_P = 1.0               # Profile weight in fused score
LAMBDA_C = 0.5               # Descent consistency weight
W_BUDGET = 25                # Influence budget window
R_MAX = 1638                 # Floor((2^15 - 1) / 20) in Z_2^16
W_CAP_MULT = 1.1             # w_cap = 1.1 / |S_benign|



# ======================================================================


# 3. CIFAR-10 Dataset & Non-IID Dirichlet Partitioning (alpha = 0.3)

def prepare_cifar10_datasets():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    print("Downloading CIFAR-10 dataset...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    return trainset, testset

# Backdoor Trigger Pattern: 4x4 white square in bottom-right corner (rows 28:32, cols 28:32)
TRIGGER_CLASS = 0 # Target label: 'airplane'
TRIGGER_NORM_VALUE = 2.5 # Saturated pixel value under normalization

class PoisonedDataset(torch.utils.data.Dataset):
    """Injects 4x4 backdoor trigger into images and sets label to target class."""
    def __init__(self, dataset, trigger_label=TRIGGER_CLASS, poison_fraction=0.20):
        self.dataset = dataset
        self.trigger_label = trigger_label
        self.poison_fraction = poison_fraction

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if np.random.rand() < self.poison_fraction and label != self.trigger_label:
            img = img.clone()
            # 4x4 white patch in bottom right (rows 28:32, cols 28:32)
            img[:, 28:32, 28:32] = TRIGGER_NORM_VALUE
            return img, self.trigger_label
        return img, label

def partition_dirichlet_non_iid(dataset, num_clients=20, alpha=0.3, seed=42):
    """Partitions dataset indices across clients using Dirichlet distribution."""
    np.random.seed(seed)
    targets = np.array(dataset.targets)
    num_classes = 10
    client_indices = [[] for _ in range(num_clients)]
    
    for k in range(num_classes):
        k_indices = np.where(targets == k)[0]
        np.random.shuffle(k_indices)
        # Sample class proportions from Dirichlet(alpha)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.array([p * (len(idx_j) < (len(dataset) / num_clients)) for p, idx_j in zip(proportions, client_indices)])
        proportions = proportions / (proportions.sum() + 1e-12)
        proportions = (np.cumsum(proportions) * len(k_indices)).astype(int)[:-1]
        
        splits = np.split(k_indices, proportions)
        for i in range(num_clients):
            client_indices[i].extend(splits[i])
            
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
        
    print(f"Dirichlet Non-IID (alpha={alpha}) partition complete for {num_clients} clients.")
    print(f"Sample sizes per client: min={min(len(idx) for idx in client_indices)}, max={max(len(idx) for idx in client_indices)}")
    return client_indices

trainset, testset = prepare_cifar10_datasets()
test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# Poisoned test loader for Backdoor Attack Success Rate (ASR) evaluation (trigger on 100% of non-target samples)
poisoned_testset = PoisonedDataset(testset, trigger_label=TRIGGER_CLASS, poison_fraction=1.0)
non_target_indices = [i for i, (_, l) in enumerate(testset) if l != TRIGGER_CLASS]
poisoned_test_loader = DataLoader(Subset(poisoned_testset, non_target_indices), batch_size=128, shuffle=False)
print("Clean and Backdoor test loaders successfully configured.")



# ======================================================================


# 4. EdgeShield-FL Defense Primitives (Pillars I - IV)
# Dedicated RNG generator isolation and FC1 activation pruning with rollback guard

def flatten_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def assign_params(model, flat_params):
    curr = 0
    for p in model.parameters():
        p_len = p.numel()
        p.data.copy_(flat_params[curr:curr + p_len].view(p.size()))
        curr += p_len

# Dedicated RNG generator for streaming Rademacher projection
# Never calls torch.manual_seed globally, preventing client noise coupling.
def streaming_rademacher_projection(update_tensor, m, seed):
    """Pseudo-random streaming Rademacher projection using dedicated generator."""
    gen = torch.Generator(device=update_tensor.device)
    gen.manual_seed(seed)
    d = update_tensor.numel()
    # Values in {-1/sqrt(m), +1/sqrt(m)}
    Phi = (torch.randint(0, 2, (m, d), generator=gen, device=update_tensor.device).float() * 2 - 1.0) / math.sqrt(m)
    return Phi @ update_tensor

def compute_layer_profile(update_tensor, layer_slices):
    """Computes the 4-layer scale-invariant profile ||g_layer|| / sum(||g_layer||)."""
    norms = []
    for (start, end) in layer_slices:
        seg = update_tensor[start:end]
        norms.append(torch.norm(seg, p=2))
    norms_tensor = torch.stack(norms)
    prof = norms_tensor / (torch.sum(norms_tensor) + 1e-12)
    return prof

def quantize_int8_downlink(delta_w, layer_slices):
    """Per-layer affine INT8 downlink broadcast (Pillar IV)."""
    delta_w = torch.nan_to_num(delta_w, nan=0.0, posinf=0.0, neginf=0.0)
    recon_chunks = []
    for (start, end) in layer_slices:
        seg = delta_w[start:end]
        s_min, s_max = seg.min().item(), seg.max().item()
        diff = s_max - s_min
        if diff > 1e-7:
            scale = diff / 255.0
            val = -s_min / scale
            if math.isnan(val) or math.isinf(val):
                zp = 0
            else:
                zp = max(-128, min(127, round(val) - 128))
            q = torch.clamp(torch.round(seg / scale) + zp, -128, 127).to(torch.int8)
            deq = (q.float() - zp) * scale
        else:
            deq = seg.clone()
        recon_chunks.append(deq)
    return torch.cat(recon_chunks)

def evaluate_proxy_accuracy(model, proxy_loader):
    """Evaluates accuracy on proxy validation set."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in proxy_loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return (correct / total) * 100.0 if total > 0 else 0.0

# Activation fine-pruning on FC1 units with a 2.0% validation accuracy rollback guard
def get_triggered_proxy_loader(proxy_loader):
    # proxy_loader dataset is a Subset of testset
    proxy_subset = proxy_loader.dataset
    # create a poisoned version
    poisoned_proxy = PoisonedDataset(proxy_subset.dataset, trigger_label=TRIGGER_CLASS, poison_fraction=1.0)
    # subset with same indices but only non-target
    # Actually, proxy_subset has indices. We want the non-target ones among those indices
    non_target_indices = [i for i in proxy_subset.indices if proxy_subset.dataset[i][1] != TRIGGER_CLASS]
    triggered_subset = torch.utils.data.Subset(poisoned_proxy, non_target_indices)
    return DataLoader(triggered_subset, batch_size=64, shuffle=False)

def execute_pillar4(model, proxy_train_loader, proxy_val_loader=None, prune_frac=0.15, do_finetune=True, arm_name="Unknown"):
    if proxy_val_loader is None:
        proxy_val_loader = proxy_train_loader
    state_backup = copy.deepcopy(model.state_dict())
    
    # 1. Evaluate Baseline on Held-out Validation Proxy Set (200 samples)
    acc_clean_pre = evaluate_proxy_accuracy(model, proxy_val_loader)
    triggered_loader = get_triggered_proxy_loader(proxy_val_loader)
    acc_trig_pre = evaluate_proxy_accuracy(model, triggered_loader)
    
    # 2. Pruning (activations measured on proxy_train_loader: 300 samples)
    model.eval()
    activations = []
    def hook_fn(module, input, output):
        activations.append(output.detach())
    
    hook = model.fc1.register_forward_hook(hook_fn)
    with torch.no_grad():
        for data, _ in proxy_train_loader:
            data = data.to(device)
            _ = model(data)
    hook.remove()
    
    pruned_units_str = "None"
    if len(activations) > 0:
        cat_act = torch.cat(activations, dim=0)
        unit_means = cat_act.mean(dim=0)
        k_prune = max(1, int(64 * prune_frac))
        dormant_units = torch.argsort(unit_means)[:k_prune]
        
        model.fc1.weight.data[dormant_units, :] = 0.0
        if model.fc1.bias is not None:
            model.fc1.bias.data[dormant_units] = 0.0
        pruned_units_str = str(dormant_units.tolist())
            
    # 3. Fine-tuning (trains on proxy_train_loader: 300 samples)
    if do_finetune:
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        for _ in range(2):
            for data, target in proxy_train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                out = model(data)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
                
    # 4. Re-evaluate on Held-out Validation Proxy Set (200 samples)
    acc_clean_post = evaluate_proxy_accuracy(model, proxy_val_loader)
    acc_trig_post = evaluate_proxy_accuracy(model, triggered_loader)
    
    # 5. Rollback Guard (Evaluated strictly on held-out clean set)
    rolled_back = False
    if acc_clean_post < (acc_clean_pre - 2.0):
        model.load_state_dict(state_backup)
        rolled_back = True
        
    # Atomic Logging
    import os
    try:
        with open("rollback.log", "r") as rf:
            logs = rf.read()
    except:
        logs = ""
    with open("rollback_tmp.log", "w") as tf:
        tf.write(logs + f"[{arm_name}] Pillar4 | FT={do_finetune} | Prune={prune_frac:.2f} | Units={pruned_units_str} | ")
        tf.write(f"HeldOut Clean {acc_clean_pre:.2f}->{acc_clean_post:.2f} | Trig {acc_trig_pre:.2f}->{acc_trig_post:.2f} | ")
        tf.write(f"Rollback={rolled_back}\n")
    os.replace("rollback_tmp.log", "rollback.log")


# ======================================================================


# 5. Local Client Training Routine & Attack Modules

def train_local_client(client_id, global_weights, dataset, client_indices, 
                       attack_type='none', gamma_scale=1.0, seed=0,
                       clip_c=None, batch_clip=None):
    """Executes LocalSGD with update clipping and optional adversarial attacks."""
    assert clip_c is not None, f"clip_c must be explicitly specified for client {client_id}"
    assert batch_clip is not None, f"batch_clip must be explicitly specified for client {client_id}"
    # Client-specific seed for DataLoader shuffling to prevent cross-client coupling
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = EdgeCNN().to(device)
    assign_params(model, global_weights)
    model.train()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=LR_LOCAL, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    subset_indices = client_indices[client_id]
    client_dataset = Subset(dataset, subset_indices)
    
    # Backdoor adversary: poison fraction gamma = 0.20
    if attack_type == 'backdoor':
        client_dataset = PoisonedDataset(client_dataset, trigger_label=TRIGGER_CLASS, poison_fraction=0.20)
        
    loader = DataLoader(client_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    
    for epoch in range(LOCAL_EPOCHS):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=batch_clip)
            optimizer.step()
            
    local_weights = flatten_params(model)
    local_weights = torch.nan_to_num(local_weights, nan=0.0, posinf=1.0, neginf=-1.0)
    update = local_weights - global_weights # g_i
    update = torch.nan_to_num(update, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Adversarial Modifications
    if attack_type == 'sign_flip':
        # 3x Byzantine Sign-Flip
        update = -3.0 * update
    elif attack_type == 'backdoor':
        # Model replacement scaling
        update = gamma_scale * update
    elif attack_type == 'gaussian_noise':
        # Byzantine Gaussian Noise
        update = torch.randn_like(update) * update.norm(p=2)
        
    update = torch.nan_to_num(update, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Update-level L2 clipping C
    u_norm = torch.norm(update, p=2)
    g_clipped = update * min(1.0, clip_c / (u_norm.item() + 1e-12))
    return torch.nan_to_num(g_clipped, nan=0.0, posinf=1.0, neginf=-1.0), u_norm.item()



# ======================================================================


# 6. EdgeShield-FL Server Anomaly Detection & Ring Aggregation
# Dual-channel aggregation and anomaly scoring:
#   - rho_s: Exact 12-NN sum of (1 - C_ij) over k = floor(0.6*N)
#   - rho_p: Exact max over layers of per-layer robust Z-score
#   - rho_c: Sketch-space cross-round projection
#   - Protocol exclusion: w_i = 0 for non-admitted clients

class EdgeShieldServer:
    def __init__(self, d_params, layer_slices, tightness_window=W_BUDGET, ema_reference=False, num_attackers=None, beta=None, seed=42):
        self.ema_reference = ema_reference
        self.d = d_params
        self.layer_slices = layer_slices
        self.tightness_window = tightness_window
        self.prev_delta_global = None
        self.num_attackers = num_attackers
        self.beta = beta
        self.eviction_seed = seed

    def aggregate_round(self, round_idx, client_uploads, defense_mode='edgeshield'):
        N = len(client_uploads)
        
        # Mode 1: Baseline FedAvg (Cleartext uniform averaging)
        if defense_mode == 'fedavg':
            stacked = torch.stack([u['g_clipped'] for u in client_uploads])
            delta_w = torch.mean(stacked, dim=0)
            self.prev_delta_global = delta_w.clone()
            return delta_w, list(range(N)), np.ones(N)/N, np.zeros(N), np.zeros(N), np.zeros(N), np.eye(N), np.zeros(N), [False]*N
            
        # Mode 2: Baseline SecAgg (Uniform averaging with dynamic norm cap, no geometric filter)
        if defense_mode == 'secagg':
            norms = [torch.norm(u['g_clipped'], p=2).item() for u in client_uploads]
            M_norm = np.median(norms)
            capped_updates = [u['g_clipped'] * min(1.0, M_norm / (torch.norm(u['g_clipped'], p=2).item() + 1e-12)) 
                              for u in client_uploads]
            stacked = torch.stack(capped_updates)
            delta_w = torch.mean(stacked, dim=0)
            self.prev_delta_global = delta_w.clone()
            return delta_w, list(range(N)), np.ones(N)/N, np.zeros(N), np.zeros(N), np.zeros(N), np.eye(N), np.zeros(N), [False]*N

        # Mode 4: Baseline Trimmed Mean (Coordinate-wise trimming of top & bottom k elements)
        # Telemetry note: Coordinate-wise robust aggregation rules operate per-coordinate;
        # client-level admission/eviction is mathematically undefined (a client's update may be trimmed in
        # some dimensions and retained in others). By formal convention, admitted is logged as list(range(N))
        # and weights as uniform np.ones(N)/N. Trimmed mean does not perform discrete client eviction.
        if defense_mode in ['trimmed_mean', 'trim']:
            stacked = torch.stack([u['g_clipped'] for u in client_uploads])
            # Derive k_trim from beta * N if beta specified, else num_attackers if > 0, else default to 0.20 * N
            if self.beta is not None and self.beta > 0:
                k_trim = int(round(self.beta * N))
            elif self.num_attackers is not None and self.num_attackers > 0:
                k_trim = self.num_attackers
            else:
                k_trim = int(round(0.20 * N))
            assert k_trim > 0 and 2 * k_trim < N, f"Invalid k_trim={k_trim} for N={N}"
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_vals[k_trim : N - k_trim]
            delta_w = torch.mean(trimmed, dim=0)
            self.prev_delta_global = delta_w.clone()
            return delta_w, list(range(N)), np.ones(N)/N, np.zeros(N), np.zeros(N), np.zeros(N), np.eye(N), np.zeros(N), [False]*N

        # Mode 5: Baseline Coordinate Median (Coordinate-wise median across clients)
        # Telemetry note: Coordinate median operates independently across each coordinate dimension;
        # per-client admission/eviction is mathematically undefined. By formal convention, admitted is logged as
        # list(range(N)) and weights as uniform np.ones(N)/N.
        # Note: torch.median on even N=20 computes coordinate-wise lower median (indices (N-1)//2).
        if defense_mode in ['median', 'coord_median', 'coordinate_median']:
            stacked = torch.stack([u['g_clipped'] for u in client_uploads])
            delta_w = torch.median(stacked, dim=0).values
            self.prev_delta_global = delta_w.clone()
            return delta_w, list(range(N)), np.ones(N)/N, np.zeros(N), np.zeros(N), np.zeros(N), np.eye(N), np.zeros(N), [False]*N

        # Mode 7: Baseline Multi-Krum (Blanchard et al., NeurIPS 2017)
        # Full-vector pairwise Euclidean distance based geometric filtering.
        # For N clients and f adversaries, score s_i is sum of closest (N - f - 2) distances.
        # Iteratively selects m = N - f clients with minimum scores and averages them.
        # Operational Parameterization (K3): In realistic deployment, the aggregator operates with
        # an assumed adversary bound f = round(beta * N) = 4 (m = 16) across both attacked and
        # clean conditions, measuring the realistic eviction penalty of distance filtering.
        if defense_mode in ['multikrum', 'multi_krum', 'krum']:
            stacked = torch.stack([u['g_clipped'] for u in client_uploads])
            if hasattr(self, 'krum_f') and self.krum_f is not None:
                f_att = self.krum_f
            elif self.beta is not None and self.beta > 0:
                f_att = int(round(self.beta * N))
            elif self.num_attackers is not None and self.num_attackers > 0:
                f_att = self.num_attackers
            else:
                f_att = int(round(0.20 * N))
            k_neighbors = max(1, N - f_att - 2)
            m_select = max(1, N - f_att)

            flat_updates = stacked.view(N, -1)
            sq_norms = torch.sum(flat_updates**2, dim=1, keepdim=True)
            dist_matrix = sq_norms + sq_norms.t() - 2.0 * (flat_updates @ flat_updates.t())
            dist_matrix = torch.clamp(dist_matrix, min=0.0)

            eye_mask = torch.eye(N, dtype=torch.bool, device=stacked.device)
            dist_matrix.masked_fill_(eye_mask, float('inf'))

            sorted_dists, _ = torch.sort(dist_matrix, dim=1)
            scores = torch.sum(sorted_dists[:, :k_neighbors], dim=1).cpu().numpy()

            selected_indices = []
            remaining = list(range(N))
            for _ in range(m_select):
                best_idx = min(remaining, key=lambda idx: scores[idx])
                selected_indices.append(best_idx)
                remaining.remove(best_idx)

            selected_indices = sorted(selected_indices)
            delta_w = torch.mean(stacked[selected_indices], dim=0)
            self.prev_delta_global = delta_w.clone()

            weights = np.zeros(N)
            for idx in selected_indices:
                weights[idx] = 1.0 / len(selected_indices)

            return delta_w, selected_indices, weights, np.zeros(N), np.zeros(N), np.zeros(N), np.eye(N), np.zeros(N), [False]*N

        # Mode 6: Random Eviction Null Control (Uniform blind eviction from honest clients only)
        # To isolate detection vs. weight-redistribution, eviction is drawn
        # strictly from honest clients (indices 0 to N - num_attackers - 1).
        # Attackers (indices N - num_attackers to N - 1) are guaranteed to be admitted in 100% of rounds.
        if defense_mode in ['random_evict', 'random_eviction']:
            num_att = self.num_attackers if self.num_attackers is not None else int(round((self.beta if self.beta is not None else 0.20) * N))
            num_honest = N - num_att
            assert num_honest >= num_att, f"num_honest ({num_honest}) < num_att ({num_att})"
            rng = np.random.default_rng(round_idx * 1000 + self.eviction_seed)
            evicted = set(rng.choice(range(num_honest), size=num_att, replace=False))
            # Verification: all evicted clients must be strictly honest (< num_honest)
            assert all(idx < num_honest for idx in evicted), f"Evicted set contains non-honest clients: {evicted}"
            admitted = [i for i in range(N) if i not in evicted]
            # Verification: all attackers must be admitted
            for att_idx in range(num_honest, N):
                assert att_idx in admitted, f"Attacker {att_idx} was erroneously evicted in control mode!"
            
            weights = np.zeros(N)
            for i in admitted:
                weights[i] = 1.0 / len(admitted)
            stacked_admitted = torch.stack([client_uploads[i]['g_clipped'] for i in admitted])
            delta_w = torch.mean(stacked_admitted, dim=0)
            
            # Telemetry tracking: compute sketch CUSUM for un-evicted/all clients
            sketches = [torch.nan_to_num(u['s_hat']).cpu().numpy() for u in client_uploads]
            S = np.nan_to_num(np.array(sketches))
            self.S_current = S
            S_norm = np.nan_to_num(S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12))
            Cm = np.nan_to_num(S_norm @ S_norm.T)
            
            if not hasattr(self, 'cusum_states') or self.cusum_states is None:
                self.cusum_states = np.zeros(N)
            if not hasattr(self, 'ema_delta_global') or self.ema_delta_global is None:
                self.ema_delta_global = None
                self.ema_count = 0
            
            # Update EMA from prev_delta_global if available
            if self.prev_delta_global is not None:
                if self.ema_delta_global is None:
                    self.ema_delta_global = self.prev_delta_global.clone().float()
                    self.ema_count = 1
                else:
                    alpha = 1.0 / 20.0 if self.ema_reference else 1.0
                    self.ema_delta_global = (1 - alpha) * self.ema_delta_global + alpha * self.prev_delta_global.clone().float()
                    self.ema_count += 1
            
            ref_delta = self.ema_delta_global if self.ema_reference else self.prev_delta_global
            if ref_delta is not None and round_idx >= 5:
                s_prev = streaming_rademacher_projection(ref_delta, m=M_SKETCH, seed=round_idx*100).cpu().numpy()
                s_prev = np.nan_to_num(s_prev)
                s_prev_norm = s_prev / (np.linalg.norm(s_prev) + 1e-12)
                rho_c = np.nan_to_num(-np.array([s_norm @ s_prev_norm for s_norm in S_norm]))
                drift = (rho_c - 0.01) / 0.1474
                self.cusum_states = np.maximum(0.0, self.cusum_states + drift)
            else:
                rho_c = np.zeros(N)
                
            # Store current delta_w for next round's reference
            self.prev_delta_global = delta_w.clone()
            return delta_w, admitted, weights, np.zeros(N), np.zeros(N), rho_c, Cm, self.cusum_states.copy(), [False]*N

        # Mode 3: EdgeShield-FL Hardened Architecture
        sketches = [torch.nan_to_num(u['s_hat']).cpu().numpy() for u in client_uploads]
        profiles = [torch.nan_to_num(u['prof_hat']).cpu().numpy() for u in client_uploads]
        norms = [float(np.nan_to_num(u['M_hat'], nan=0.0)) for u in client_uploads]
        
        S = np.nan_to_num(np.array(sketches)) # [N, m]
        self.S_current = S
        S_norm = np.nan_to_num(S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12))
        Cm = np.nan_to_num(S_norm @ S_norm.T) # Pairwise cosine matrix
        
        # Spatial score rho_s = Sum of (1 - C_ij) over k = floor(0.6*N) = 12 nearest neighbours
        k = max(1, int(0.6 * N))
        Cm_temp = Cm.copy()
        np.fill_diagonal(Cm_temp, -np.inf)
        nn = np.argsort(-Cm_temp, axis=1)[:, :k]
        rho_s = np.nan_to_num((1.0 - Cm[np.arange(N)[:, None], nn]).sum(axis=1))
        
        # Layer profile score rho_p = Max over layers of per-layer robust Z-score
        P = np.nan_to_num(np.array(profiles)) # [N, L]
        med_p = np.median(P, axis=0)
        mad_p = 1.4826 * np.median(np.abs(P - med_p), axis=0) + 1e-9
        rho_p = np.nan_to_num(np.max(np.abs(P - med_p) / mad_p, axis=1))
        
        # Tightness tracker for collusion: mutual pair-crossing
        # Note: Cm is symmetric by construction (S_norm @ S_norm.T), so Cm[i, j] == Cm[j, i].
        # We evaluate distinct pairs (i < j) to evict both endpoints of any colluding pair.
        TAU_TIGHT = 0.72
        colluding_mask = np.zeros(N, dtype=bool)
        for i in range(N):
            for j in range(i + 1, N):
                if Cm[i, j] > TAU_TIGHT:
                    colluding_mask[i] = True
                    colluding_mask[j] = True
        colluding = colluding_mask

        # Initialize CUSUM states if not present
        if not hasattr(self, 'cusum_states') or self.cusum_states is None:
            self.cusum_states = np.zeros(N)

        # CUSUM(rho_c) in sketch space
        if not hasattr(self, 'ema_delta_global') or self.ema_delta_global is None:
            self.ema_delta_global = None
            self.ema_count = 0

        # Update EMA if prev_delta_global is available
        if self.prev_delta_global is not None:
            if self.ema_delta_global is None:
                self.ema_delta_global = self.prev_delta_global.clone().float()
                self.ema_count = 1
            else:
                alpha = 1.0 / 20.0 if self.ema_reference else 1.0
                self.ema_delta_global = (1 - alpha) * self.ema_delta_global + alpha * self.prev_delta_global.clone().float()
                self.ema_count += 1

        ref_delta = self.ema_delta_global if self.ema_reference else self.prev_delta_global

        if ref_delta is not None and round_idx >= 5:
            s_prev = streaming_rademacher_projection(ref_delta, m=M_SKETCH, seed=round_idx*100).cpu().numpy()
            s_prev = np.nan_to_num(s_prev)
            s_prev_norm = s_prev / (np.linalg.norm(s_prev) + 1e-12)
            rho_c = np.nan_to_num(-np.array([s_norm @ s_prev_norm for s_norm in S_norm]))
            
            # Update CUSUM (Frozen parameters: nu=0.01, sigma=0.1474, h=5)
            CUSUM_NU = 0.01
            CUSUM_SIGMA = 0.1474
            drift = (rho_c - CUSUM_NU) / CUSUM_SIGMA
            self.cusum_states = np.maximum(0.0, self.cusum_states + drift)
        else:
            rho_c = np.zeros(N)

        # State-gated weights
        CUSUM_H = 5.0
        W_FLOOR = 0.01
        weights = np.zeros(N)
        
        floored = (self.cusum_states >= CUSUM_H) & ~colluding
        cap = ~colluding & ~floored
        num_cap = cap.sum()
        
        if num_cap == 0:
            remaining = floored.sum()
            if remaining > 0:
                weights[floored] = 1.0 / remaining
        else:
            weights[floored] = W_FLOOR
            weight_remaining = 1.0 - (floored.sum() * W_FLOOR)
            if weight_remaining < 0:
                weights[floored] = 1.0 / floored.sum()
            else:
                weights[cap] = weight_remaining / num_cap
                
        admitted = [i for i in range(N) if weights[i] > 0]
        
        # Assertions
        if len(admitted) > 0:
            assert np.isclose(weights.sum(), 1.0), f"Weights sum to {weights.sum()}"
            if num_cap > 0:
                w_cap_limit = 1.1 / num_cap
                for i in admitted:
                    if cap[i]:
                        assert weights[i] <= w_cap_limit + 1e-5, f"Weight {weights[i]} exceeds cap {w_cap_limit}"


        # Dynamic norm bounding M_norm over all non-colluding
        non_colluding_norms = [norms[i] for i in range(N) if not colluding[i]]
        if len(non_colluding_norms) == 0:
            M_norm = 1e-6
        else:
            M_norm = max(1e-6, float(np.median(non_colluding_norms)))
        
        # Stochastic Quantization in Z_2^16 & Masked Sum (with division-by-zero guards)
        num_adm = len(admitted)
        w_cap = (W_CAP_MULT / num_cap) if num_cap > 0 else (1.0 / max(1, num_adm))
        Delta_global = max(1e-9, (w_cap * M_norm) / R_MAX)
        weighted_updates = []
        for i in admitted:
            g = torch.nan_to_num(client_uploads[i]['g_clipped'])
            g_norm = torch.norm(g, p=2).item()
            g_capped = g * min(1.0, M_norm / (g_norm + 1e-12))
            u_i = weights[i] * g_capped
            # Stochastic Rounding
            u_scaled = u_i / (Delta_global + 1e-12)
            u_round = torch.floor(u_scaled + torch.rand_like(u_scaled))
            u_clamped = torch.clamp(u_round, -R_MAX, R_MAX)
            weighted_updates.append(u_clamped)
            
        recon_sum = torch.nan_to_num(torch.stack(weighted_updates).sum(dim=0)) if num_adm > 0 else torch.zeros(self.d, device=device)
        delta_w_clean = torch.nan_to_num(recon_sum * Delta_global)
        
        # Pillar IV: Dilution Noise
        sigma_coord = (0.5 * M_norm) / math.sqrt(max(1, num_adm) * self.d)
        dilution = torch.randn_like(delta_w_clean) * sigma_coord
        delta_w_global = torch.nan_to_num(delta_w_clean + dilution)
        
        # INT8 Downlink Compression
        delta_w_quantized = quantize_int8_downlink(delta_w_global, self.layer_slices)
        self.prev_delta_global = delta_w_quantized.clone()
        return delta_w_quantized, admitted, weights, rho_s, rho_p, rho_c, Cm, self.cusum_states.copy(), colluding.tolist()



# ======================================================================


# 7. Model Evaluation Function (Clean Accuracy & Backdoor ASR)

def evaluate_model(model, clean_loader, backdoor_loader):
    model.eval()
    correct_clean, total_clean = 0, 0
    with torch.no_grad():
        for data, target in clean_loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            correct_clean += (pred == target).sum().item()
            total_clean += target.size(0)
    clean_acc = (correct_clean / total_clean) * 100.0 if total_clean > 0 else 0.0
    
    # Backdoor Attack Success Rate (ASR)
    correct_backdoor, total_backdoor = 0, 0
    with torch.no_grad():
        for data, target in backdoor_loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            correct_backdoor += (pred == target).sum().item()
            total_backdoor += target.size(0)
    backdoor_asr = (correct_backdoor / total_backdoor) * 100.0 if total_backdoor > 0 else 0.0
    return clean_acc, backdoor_asr



# ======================================================================


# 8. Federated Simulation Runner for an Experimental Arm
# Per-round client telemetry and independent per-client noise draws

def run_federated_arm(arm_name, rounds=100, beta=0.20, defense_mode='edgeshield', 
                      attack_type='none', seed=1, k_prune=None, ft_on=None,
                      clip_c=None, batch_clip=None, ema_reference=False, num_attackers=4, dirichlet_alpha=0.3):
    """Executes a complete federated learning arm with full empirical logging."""
    assert k_prune is not None, f"k_prune must be explicitly specified for arm {arm_name}"
    assert ft_on is not None, f"ft_on must be explicitly specified for arm {arm_name}"
    assert clip_c is not None, f"clip_c must be explicitly specified for arm {arm_name}"
    assert batch_clip is not None, f"batch_clip must be explicitly specified for arm {arm_name}"
    if attack_type != 'none':
        assert num_attackers == int(round(beta * NUM_CLIENTS)), f"Config identity violation: num_attackers={num_attackers} != beta*{NUM_CLIENTS}"
    else:
        assert num_attackers == 0, f"Clean arm {arm_name} must have num_attackers=0, got {num_attackers}"
    assert dirichlet_alpha is not None, f"dirichlet_alpha must be explicitly specified for arm {arm_name}"
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\n{'='*75}")
    print(f"STARTING ARM: [{arm_name}]")
    print(f"CONFIG ASSERTION: Defense={defense_mode} | Attack={attack_type} | Rounds={rounds} | Beta={beta*100:.0f}% | Seed={seed}")
    print(f"CONFIG ASSERTION: Clip_C={clip_c} | BatchClip={batch_clip} | K_prune={k_prune} | FT_on={ft_on} | EMA={ema_reference} | NumAtt={num_attackers} | Alpha={dirichlet_alpha}")
    print(f"{'='*75}")
    
    client_indices = partition_dirichlet_non_iid(trainset, num_clients=NUM_CLIENTS, alpha=dirichlet_alpha, seed=seed)
    proxy_train_subset = Subset(testset, list(range(300)))
    proxy_val_subset = Subset(testset, list(range(300, 500)))
    proxy_train_loader = DataLoader(proxy_train_subset, batch_size=64, shuffle=False)
    proxy_val_loader = DataLoader(proxy_val_subset, batch_size=64, shuffle=False)
    
    global_model = EdgeCNN().to(device)
    layer_slices = global_model.get_layer_slices()
    server = EdgeShieldServer(
        d_params=136874, 
        layer_slices=layer_slices, 
        tightness_window=W_BUDGET, 
        ema_reference=ema_reference, 
        num_attackers=num_attackers, 
        beta=beta,
        seed=seed
    )
    
    num_byz = num_attackers
    byz_clients = set(range(NUM_CLIENTS - num_byz, NUM_CLIENTS)) if num_byz > 0 else set()
    print(f"Adversarial client IDs: {sorted(list(byz_clients)) if byz_clients else 'None'}")
    
    history = {
        'round': [],
        'is_eval_round': [],
        'clean_acc': [],
        'backdoor_asr': [],
        'round_time_s': [],
        'admitted': [],
        'weights': [],
        'rho_s': [],
        'rho_p': [],
        'rho_c': [],
        'pairwise_cos_matrix': [],
        'unclipped_norms': [],
        'delta_cos_prev': [],
        'cusum_states': [],
        'colluding': []
    }
    
    # Zero-round evaluation
    init_acc, init_asr = evaluate_model(global_model, test_loader, poisoned_test_loader)
    print(f"Round 0 (Untrained): Clean Acc = {init_acc:.2f}%, Backdoor ASR = {init_asr:.2f}%")
    history['round'].append(0)
    history['is_eval_round'].append(True)
    history['clean_acc'].append(init_acc)
    history['backdoor_asr'].append(init_asr)
    history['round_time_s'].append(0.0)
    history['admitted'].append(list(range(NUM_CLIENTS)))
    history['weights'].append([1.0/NUM_CLIENTS]*NUM_CLIENTS)
    history['rho_s'].append([0.0]*NUM_CLIENTS)
    history['rho_p'].append([0.0]*NUM_CLIENTS)
    history['rho_c'].append([0.0]*NUM_CLIENTS)
    history['pairwise_cos_matrix'].append(np.eye(NUM_CLIENTS).tolist())
    history['unclipped_norms'].append([0.0]*NUM_CLIENTS)
    history['raw_sketches'] = [[]]
    history['ema_count'] = [0]
    history['delta_cos_prev'].append(1.0)
    history['cusum_states'].append([0.0]*NUM_CLIENTS)
    history['colluding'].append([False]*NUM_CLIENTS)
    
    prev_delta_global = None
    pbar = tqdm(range(1, rounds + 1), desc=f"Arm: {arm_name[:22]}")
    for t in pbar:
        t0 = time.time()
        global_weights = flatten_params(global_model)
        
        # 1. Edge Client Phase 1: LocalSGD & Obfuscated Inspection
        client_uploads = []
        rd_unclipped_norms = []
        for i in range(NUM_CLIENTS):
            is_byz = (i in byz_clients)
            client_attack = attack_type if is_byz else 'none'
            
            # Local update
            if is_byz and attack_type == 'mimicry':
                if prev_delta_global is not None:
                    g_i = prev_delta_global.clone().float()
                else:
                    g_i = torch.zeros(136874, device=global_weights[0].device)
                unclipped_norm = torch.norm(g_i, p=2).item()
                g_i = g_i * min(1.0, clip_c / (unclipped_norm + 1e-12))
            else:
                g_i, unclipped_norm = train_local_client(
                    client_id=i,
                    global_weights=global_weights,
                    dataset=trainset,
                    client_indices=client_indices,
                    attack_type=client_attack,
                    gamma_scale=20.0 if client_attack == 'backdoor' else 1.0,
                    seed=seed * 100000 + t * 1000 + i,
                    clip_c=clip_c,
                    batch_clip=batch_clip
                )
            g_i = torch.nan_to_num(g_i)
            rd_unclipped_norms.append(unclipped_norm)
            
            # Dedicated RNG generator for independent per-client noise
            c_gen = torch.Generator(device=g_i.device)
            c_gen.manual_seed(seed * 100000 + t * 1000 + i)
            
            # Shared projection matrix across clients in round t
            s_i = streaming_rademacher_projection(g_i, m=M_SKETCH, seed=t*100)
            # Independent sketch noise
            s_hat_i = s_i + torch.randn(s_i.shape, generator=c_gen, device=g_i.device) * SIGMA_S
            
            prof_i = compute_layer_profile(g_i, layer_slices)
            # Independent profile noise
            prof_hat_i = prof_i + torch.randn(prof_i.shape, generator=c_gen, device=g_i.device) * SIGMA_PROF
            
            norm_true = torch.norm(g_i, p=2).item()
            lap_noise = np.random.default_rng(seed * 100000 + t * 1000 + i).laplace(0, B_M_LAPLACE)
            M_hat_i = max(0.0, float(norm_true + lap_noise))
            
            client_uploads.append({
                'g_clipped': g_i,
                's_hat': s_hat_i,
                'prof_hat': prof_hat_i,
                'M_hat': M_hat_i
            })
            
        # 2. Server Phase 2: Tri-Modal Filtering & Ring Aggregation
        delta_w_global, admitted, weights, rho_s, rho_p, rho_c, Cm, cusum_st, colluding_flags = server.aggregate_round(
            round_idx=t,
            client_uploads=client_uploads,
            defense_mode=defense_mode
        )
        
        # 3. Apply Update to Global Model
        new_weights = global_weights + delta_w_global
        new_weights = torch.nan_to_num(new_weights)
        assign_params(global_model, new_weights)
        
        # Directional correlation with previous global update
        if prev_delta_global is not None:
            dot_prod = torch.dot(delta_w_global, prev_delta_global).item()
            norm1 = delta_w_global.norm(p=2).item()
            norm2 = prev_delta_global.norm(p=2).item()
            cos_dir = dot_prod / (norm1 * norm2 + 1e-12)
        else:
            cos_dir = 1.0
        prev_delta_global = delta_w_global.clone()
        history['delta_cos_prev'].append(float(cos_dir))
        
        # 4. Periodic Activation Fine-Pruning + Proxy Distillation (Guarded on 200 held-out samples)
        if defense_mode == 'edgeshield' and t % k_prune == 0:
            execute_pillar4(global_model, proxy_train_loader, proxy_val_loader, prune_frac=0.15, do_finetune=ft_on, arm_name=arm_name)
            
        elapsed = time.time() - t0
        
        # Per-round history logging
        is_eval = (t % 5 == 0 or t == rounds)
        history['round'].append(t)
        history['is_eval_round'].append(is_eval)
        history['admitted'].append(admitted)
        history['weights'].append(weights.tolist())
        history['rho_s'].append(rho_s.tolist())
        history['rho_p'].append(rho_p.tolist())
        history['rho_c'].append(rho_c.tolist())
        history['pairwise_cos_matrix'].append(Cm.tolist())
        history['unclipped_norms'].append(rd_unclipped_norms)
        history['round_time_s'].append(elapsed)
        history['cusum_states'].append(cusum_st.tolist() if hasattr(cusum_st, 'tolist') else list(cusum_st))
        
        # Log raw sketches for offline tuning if EMA is active
        if hasattr(server, 'S_current') and server.S_current is not None:
            history['raw_sketches'].append(server.S_current.tolist())
        else:
            history['raw_sketches'].append([])
            
        history['ema_count'].append(server.ema_count if hasattr(server, 'ema_count') else 0)
        history['colluding'].append(colluding_flags)
        
        # Evaluate accuracy every 5 rounds or at final round
        if is_eval:
            clean_acc, backdoor_asr = evaluate_model(global_model, test_loader, poisoned_test_loader)
            history['clean_acc'].append(clean_acc)
            history['backdoor_asr'].append(backdoor_asr)
            pbar.set_postfix({'Acc': f"{clean_acc:.2f}%", 'ASR': f"{backdoor_asr:.2f}%", 'Sec/Rnd': f"{elapsed:.1f}s"})
            print(f"  [Round {t:3d}/{rounds}] Clean Acc: {clean_acc:6.2f}% | ASR: {backdoor_asr:6.2f}% | Round Time: {elapsed:4.1f}s", flush=True)
        else:
            # Observation 2: Explicit None on non-eval rounds to prevent duplicate-value ambiguity in JSON
            history['clean_acc'].append(None)
            history['backdoor_asr'].append(None)
            
    final_clean = [a for a in history['clean_acc'] if a is not None][-1]
    final_asr = [a for a in history['backdoor_asr'] if a is not None][-1]
    print(f"Finished [{arm_name}]: Final Clean Acc = {final_clean:.2f}%, Backdoor ASR = {final_asr:.2f}%")
    return history


def run_experiment(defense_mode, attack_type, rounds=100, seed=42, arm_name=None, 
                   k_prune=None, ft_on=None, beta=None, clip_c=None, batch_clip=None, ema_reference=False, num_attackers=4, dirichlet_alpha=0.3):
    """Convenience wrapper for running benchmark arms with explicit configuration."""
    if arm_name is None:
        arm_name = f"{defense_mode}_{attack_type}_seed{seed}"
    if beta is None:
        beta = 0.20 if attack_type == 'sign_flip' else (0.05 if attack_type == 'backdoor' else 0.0)
    assert k_prune is not None, f"k_prune must be explicitly specified in run_experiment for arm {arm_name}"
    assert ft_on is not None, f"ft_on must be explicitly specified in run_experiment for arm {arm_name}"
    assert clip_c is not None, f"clip_c must be explicitly specified in run_experiment for arm {arm_name}"
    assert batch_clip is not None, f"batch_clip must be explicitly specified in run_experiment for arm {arm_name}"
    return run_federated_arm(
        arm_name=arm_name,
        rounds=rounds,
        beta=beta,
        defense_mode=defense_mode,
        attack_type=attack_type,
        seed=seed,
        k_prune=k_prune,
        ft_on=ft_on,
        clip_c=clip_c,
        batch_clip=batch_clip,
        ema_reference=ema_reference,
        num_attackers=num_attackers,
        dirichlet_alpha=dirichlet_alpha
    )



# ======================================================================

# ======================================================================
# 9. Stage 7 Milestone 1: Alpha=1.0 5-Seed Confirmation (Seeds 45 & 46) - ~6.5h
# Arms: 6 arms total across seeds [45, 46]
# Hardware expectation: ~6.6 GPU-hours on Kaggle P100 / T4
# Note: 6.6 GPU-hrs was the pre-run planning estimate; actual completed wall-clock compute recorded in Table 8 was 5.5 GPU-hrs.

STAGE7_CONFIGS = [
    {'name': 'FedAvg_Clean_a1.0_s45', 'defense': 'fedavg', 'attack': 'none', 'seed': 45, 'k_prune': 25, 'ft': False, 'ema': False, 'num_att': 0, 'alpha': 1.0, 'beta': 0.0},
    {'name': 'FedAvg_SignFlip20_a1.0_s45', 'defense': 'fedavg', 'attack': 'sign_flip', 'seed': 45, 'k_prune': 25, 'ft': False, 'ema': False, 'num_att': 4, 'alpha': 1.0, 'beta': 0.20},
    {'name': 'EdgeShield_EMA_SignFlip20_a1.0_s45', 'defense': 'edgeshield', 'attack': 'sign_flip', 'seed': 45, 'k_prune': 25, 'ft': False, 'ema': True, 'num_att': 4, 'alpha': 1.0, 'beta': 0.20},
    {'name': 'FedAvg_Clean_a1.0_s46', 'defense': 'fedavg', 'attack': 'none', 'seed': 46, 'k_prune': 25, 'ft': False, 'ema': False, 'num_att': 0, 'alpha': 1.0, 'beta': 0.0},
    {'name': 'FedAvg_SignFlip20_a1.0_s46', 'defense': 'fedavg', 'attack': 'sign_flip', 'seed': 46, 'k_prune': 25, 'ft': False, 'ema': False, 'num_att': 4, 'alpha': 1.0, 'beta': 0.20},
    {'name': 'EdgeShield_EMA_SignFlip20_a1.0_s46', 'defense': 'edgeshield', 'attack': 'sign_flip', 'seed': 46, 'k_prune': 25, 'ft': False, 'ema': True, 'num_att': 4, 'alpha': 1.0, 'beta': 0.20},
]

import os
import json
import math

results_file = 'stage7_alpha1_5seeds_results.json'
results_data = {}

# Guarded resume read from stage7_alpha1_5seeds_results.json
if os.path.exists(results_file):
    try:
        with open(results_file, 'r') as rf:
            loaded = json.load(rf).get('results', {})
            for k, v in loaded.items():
                if len(v.get('round', [])) >= 100:
                    results_data[k] = v
        print(f"--> [RESUME] Found {len(results_data)} completed arms from {results_file}: {list(results_data.keys())}")
    except (json.JSONDecodeError, OSError) as e:
        print(f"--> [RESUME WARNING] Could not parse existing {results_file} ({e}), starting clean.")

for cfg in STAGE7_CONFIGS:
    arm_name = cfg['name']
    # Pre-execution standing identity assertions
    assert cfg['num_att'] == int(round(cfg['beta'] * 20)), f"Config identity violation in {arm_name}: num_att={cfg['num_att']} != beta*20={cfg['beta']*20}"
    assert cfg['alpha'] == 1.0, f"Config error: alpha must be 1.0 in {arm_name}"

    if arm_name in results_data and len(results_data[arm_name].get('round', [])) >= 100:
        last_clean = results_data[arm_name]['clean_acc'][-1]
        print(f"\n--> [SKIP/RESUME] Arm [{arm_name}] already completed (Clean: {last_clean:.2f}%)")
        continue

    print(f"\n{'='*75}\nSTARTING STAGE 7 ARM: [{arm_name}]\n{'='*75}")
    history = run_experiment(
        defense_mode=cfg['defense'],
        attack_type=cfg['attack'],
        seed=cfg['seed'],
        arm_name=arm_name,
        k_prune=cfg['k_prune'],
        ft_on=cfg['ft'],
        clip_c=5.0,
        batch_clip=5.0,
        ema_reference=cfg['ema'],
        num_attackers=cfg['num_att'],
        dirichlet_alpha=cfg['alpha'],
        beta=cfg['beta']
    )
    results_data[arm_name] = history
    # Hardened atomic checkpoint save via temporary file + os.replace
    tmp_file = f'{results_file}.tmp'
    with open(tmp_file, 'w') as wf:
        json.dump({'results': results_data}, wf)
    os.replace(tmp_file, results_file)
    print(f"\n--- Arm {arm_name} COMPLETE. Saved atomically to {results_file}. ---\n")


# ======================================================================
# 10. Automated Pre-Registered Statistical Analysis: Paired Confidence Intervals
# Computes within-seed paired difference D_s = Acc_EdgeShield(s) - Acc_SF(s)
# Combines Stage 6B/6C anchors (Seeds 42, 43, 44) with new completed seeds.

print("\n" + "=" * 80)
print("STAGE 7: PAIRED DIFFERENCE ANALYSIS & CONFIDENCE INTERVALS (alpha=1.0)")
print("=" * 80)

# Historical anchors from Stage 6B and Stage 6C
anchor_seeds = {
    42: {'clean': 66.94, 'sf': 59.22, 'es': 61.56},
    43: {'clean': 64.24, 'sf': 57.88, 'es': 59.98},
    44: {'clean': 66.41, 'sf': 58.11, 'es': 63.23},
}

all_seed_data = {}
for s, d in anchor_seeds.items():
    all_seed_data[s] = dict(d)

# Extract new seeds from results_data
if os.path.exists('stage7_alpha1_5seeds_results.json'):
    with open('stage7_alpha1_5seeds_results.json', 'r') as rf:
        current_results = json.load(rf).get('results', {})
else:
    current_results = results_data

# Detect completed seeds in current_results
detected_seeds = set()
for arm_name in current_results:
    if '_s' in arm_name:
        try:
            s_val = int(arm_name.split('_s')[-1])
            detected_seeds.add(s_val)
        except ValueError:
            pass

for s in sorted(detected_seeds):
    clean_key = f"FedAvg_Clean_a1.0_s{s}"
    sf_key = f"FedAvg_SignFlip20_a1.0_s{s}"
    es_key = f"EdgeShield_EMA_SignFlip20_a1.0_s{s}"

    if clean_key in current_results and sf_key in current_results and es_key in current_results:
        clean_acc = [x for x in current_results[clean_key]['clean_acc'] if x is not None][-1]
        sf_acc = [x for x in current_results[sf_key]['clean_acc'] if x is not None][-1]
        es_acc = [x for x in current_results[es_key]['clean_acc'] if x is not None][-1]
        all_seed_data[s] = {'clean': clean_acc, 'sf': sf_acc, 'es': es_acc}

print(f"Total Partitions Available for Paired Analysis: N = {len(all_seed_data)}")
print("-" * 80)
print(f"{'Seed':<6} | {'FedAvg Clean':<14} | {'FedAvg SF':<14} | {'EdgeShield':<14} | {'Deficit':<10} | {'Recovered (D)':<14} | {'R (%)':<8}")
print("-" * 80)

paired_diffs = []
recovery_ratios = []
clean_accs = []
sf_accs = []
es_accs = []

for s in sorted(all_seed_data.keys()):
    row = all_seed_data[s]
    clean = row['clean']
    sf = row['sf']
    es = row['es']
    deficit = clean - sf
    d_s = es - sf
    r_pct = (d_s / deficit) * 100.0 if deficit > 0 else 0.0

    paired_diffs.append(d_s)
    recovery_ratios.append(r_pct)
    clean_accs.append(clean)
    sf_accs.append(sf)
    es_accs.append(es)

    print(f"{s:<6} | {clean:6.2f}%        | {sf:6.2f}%        | {es:6.2f}%        | {deficit:6.2f}%   | {d_s:+6.2f}%        | {r_pct:6.2f}%")

print("-" * 80)

N = len(paired_diffs)
mean_d = sum(paired_diffs) / N
var_d = sum((x - mean_d) ** 2 for x in paired_diffs) / (N - 1) if N > 1 else 0.0
std_d = math.sqrt(var_d)
se_d = std_d / math.sqrt(N) if N > 0 else 0.0

mean_r = sum(recovery_ratios) / N
var_r = sum((x - mean_r) ** 2 for x in recovery_ratios) / (N - 1) if N > 1 else 0.0
std_r = math.sqrt(var_r)
se_r = std_r / math.sqrt(N) if N > 0 else 0.0

# Critical t-values (two-tailed 95% alpha=0.05, df = N-1)
t_critical_table = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447,  7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131
}
t_crit = t_critical_table.get(N - 1, 2.0)

ci_lower_d = mean_d - t_crit * se_d
ci_upper_d = mean_d + t_crit * se_d

ci_lower_r = mean_r - t_crit * se_r
ci_upper_r = mean_r + t_crit * se_r

t_stat = mean_d / se_d if se_d > 0 else 0.0

print(f"\nPAIRED SUMMARY STATISTICS (N = {N} Seeds):")
print(f"  • Mean Paired Recovery (D) : {mean_d:+.2f}%  (Std = {std_d:.2f}%, SE = {se_d:.2f}%)")
print(f"  • 95% Confidence Interval   : [{ci_lower_d:+.2f}%, {ci_upper_d:+.2f}%]  (t_crit = {t_crit:.3f}, df = {N-1})")
print(f"  • Paired t-statistic        : t = {t_stat:.3f}")
print(f"  • Mean Recovery Ratio (R)   : {mean_r:.2f}%  (Std = {std_r:.2f}%, SE = {se_r:.2f}%)")
print(f"  • 95% CI on Recovery Ratio  : [{ci_lower_r:.2f}%, {ci_upper_r:.2f}%]")

print("\nPRE-REGISTERED HYPOTHESIS EVALUATION:")
if ci_lower_d > 0.0:
    print("  >>> [BRANCH A CONFIRMED]: 95% CI strictly excludes zero (ci_lower > 0.0%).")
    print(f"      Statistically significant recovery door established across {N} partitions at alpha=1.0!")
else:
    print(f"  >>> [BRANCH B: UNRESOLVED]: 95% CI spans zero ([{ci_lower_d:+.2f}%, {ci_upper_d:+.2f}%]).")
    print("      Net accuracy recovery is statistically unresolved at the stochastic SGD noise floor.")
print("=" * 80 + "\n")
