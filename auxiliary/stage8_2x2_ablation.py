#!/usr/bin/env python3
"""
Stage 8: 2x2 Factorial Ablation of Sketch/CUSUM Momentum Tracking vs. Server-Side Guarded Pruning
Under Zero Test-Set Overlap (Dirichlet Alpha = 1.0, 20% Sign-Flip)

Protocol: PREREG-STAGE-8-2X2-ABLATION (SHA-256: 3eeca778616da13429aa724a0fef05c52e40ea6283a5ae159bc0bc7d7a9e45ce)
Target Artifact: stage8_2x2_ablation_results.json
Log Target: rollback_stage8.log
Checkpoints: checkpoints/stage8_{arm_name}_r100.pt

This script is fully self-contained and optimized for Kaggle/Cloud GPU execution (NVIDIA P100/T4).
Features:
  1. Zero test-set overlap: 500 proxy samples drawn from CIFAR-10 training split (0..499).
  2. Disjoint client training pool: client Dirichlet partitions drawn from trainset (500..49999).
  3. Dual evaluation: evaluates both full 10,000 test set and disjoint 9,500 test set (500..9999).
  4. Explicit round logging: rollback_stage8.log records explicit Round={t} in each entry.
  5. Terminal checkpoint saving: saves round-100 model state dict for every completed arm.
  6. Re-entrant atomic checkpointing: resumes seamlessly across Kaggle timeout boundaries.
"""

import os
import sys
import time
import math
import copy
import json
import hashlib
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

# Hardware Verification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=" * 75)
print(f"PyTorch Version: {torch.__version__}")
print(f"Execution Device: {device}")
if torch.cuda.is_available():
    print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Available VRAM:  {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: CUDA not detected! Running on CPU will be significantly slower.")
print("=" * 75)


# Model Architecture: Exact 4-layer CNN (d = 136,874 parameters)
class EdgeCNN(nn.Module):
    def __init__(self):
        super(EdgeCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(2048, 64)
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
        slices = []
        curr = 0
        for p in [list(self.conv1.parameters()), list(self.conv2.parameters()),
                  list(self.fc1.parameters()), list(self.fc2.parameters())]:
            p_len = sum(param.numel() for param in p)
            slices.append((curr, curr + p_len))
            curr += p_len
        return slices


sample_model = EdgeCNN()
d_params = sum(p.numel() for p in sample_model.parameters())
assert d_params == 136874, f"Parameter count mismatch: {d_params} != 136874"
layer_slices = sample_model.get_layer_slices()
print(f"Model validated: d = {d_params:,} parameters across 4 layers.")

# Hyperparameters
NUM_CLIENTS = 20
LOCAL_EPOCHS = 2
BATCH_SIZE = 64
LR_LOCAL = 0.08
CLIP_C = 5.0
BATCH_CLIP_NORM = 5.0
M_SKETCH = 64
SIGMA_S = 0.05
SIGMA_PROF = 0.01
B_M_LAPLACE = 0.05
W_BUDGET = 25
R_MAX = 1638
W_CAP_MULT = 1.1

# Backdoor pattern constants (for evaluation continuity)
TRIGGER_CLASS = 0
TRIGGER_NORM_VALUE = 2.5


class PoisonedDataset(torch.utils.data.Dataset):
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
            img[:, 28:32, 28:32] = TRIGGER_NORM_VALUE
            return img, self.trigger_label
        return img, label


# Prepare Datasets with Zero Test-Set Overlap
def prepare_datasets():
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

    print("Downloading/loading CIFAR-10...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # 1. Zero Overlap Server Proxy Set (Drawn strictly from trainset: 0..499)
    # 300 for activation profiling, 200 for rollback validation guard
    proxy_train_subset = Subset(trainset, list(range(0, 300)))
    proxy_val_subset = Subset(trainset, list(range(300, 500)))

    # 2. Client Training Pool (Drawn from trainset: 500..49999)
    # Strictly disjoint from server proxy set
    client_pool_indices = list(range(500, len(trainset)))

    # 3. Test Loaders: Full (10,000) and Disjoint (indices 500..9999)
    full_test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    disjoint_test_subset = Subset(testset, list(range(500, len(testset))))
    disjoint_test_loader = DataLoader(disjoint_test_subset, batch_size=128, shuffle=False, num_workers=2)

    # Poisoned test loader for continuity
    poisoned_testset = PoisonedDataset(testset, trigger_label=TRIGGER_CLASS, poison_fraction=1.0)
    non_target_indices = [i for i, (_, l) in enumerate(testset) if l != TRIGGER_CLASS]
    poisoned_test_loader = DataLoader(Subset(poisoned_testset, non_target_indices), batch_size=128, shuffle=False)

    print("Datasets partitioned:")
    print(f"  Server Proxy Train: {len(proxy_train_subset)} samples (trainset[0:300])")
    print(f"  Server Proxy Val:   {len(proxy_val_subset)} samples (trainset[300:500])")
    print(f"  Client Train Pool:  {len(client_pool_indices)} samples (trainset[500:50000])")
    print(f"  Full Test Set:      {len(testset)} samples (testset[0:10000], ZERO overlap)")
    print(f"  Disjoint Test Set:  {len(disjoint_test_subset)} samples (testset[500:10000])")

    return (trainset, testset, proxy_train_subset, proxy_val_subset,
            client_pool_indices, full_test_loader, disjoint_test_loader, poisoned_test_loader)


trainset, testset, proxy_train_subset, proxy_val_subset, client_pool_indices, full_test_loader, disjoint_test_loader, poisoned_test_loader = prepare_datasets()


def partition_dirichlet_client_pool(trainset, pool_indices, num_clients=20, alpha=1.0, seed=42):
    """Partitions pool_indices across clients using Dirichlet(alpha)."""
    np.random.seed(seed)
    targets = np.array([trainset.targets[idx] for idx in pool_indices])
    num_classes = 10
    client_indices = [[] for _ in range(num_clients)]

    for k in range(num_classes):
        k_local_idx = np.where(targets == k)[0]
        k_actual_idx = [pool_indices[i] for i in k_local_idx]
        np.random.shuffle(k_actual_idx)

        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.array([p * (len(idx_j) < (len(pool_indices) / num_clients)) for p, idx_j in zip(proportions, client_indices)])
        proportions = proportions / (proportions.sum() + 1e-12)
        proportions = (np.cumsum(proportions) * len(k_actual_idx)).astype(int)[:-1]

        splits = np.split(k_actual_idx, proportions)
        for i in range(num_clients):
            client_indices[i].extend(splits[i])

    for i in range(num_clients):
        np.random.shuffle(client_indices[i])

    print(f"Dirichlet Non-IID (alpha={alpha}, seed={seed}) partitioned over {len(pool_indices)} training samples.")
    return client_indices


# Model Utilities
def flatten_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def assign_params(model, flat_params):
    curr = 0
    for p in model.parameters():
        p_len = p.numel()
        p.data.copy_(flat_params[curr:curr + p_len].view(p.size()))
        curr += p_len


def streaming_rademacher_projection(update_tensor, m, seed):
    gen = torch.Generator(device=update_tensor.device)
    gen.manual_seed(seed)
    d = update_tensor.numel()
    Phi = (torch.randint(0, 2, (m, d), generator=gen, device=update_tensor.device).float() * 2 - 1.0) / math.sqrt(m)
    return Phi @ update_tensor


def compute_layer_profile(update_tensor, slices):
    norms = []
    for (start, end) in slices:
        seg = update_tensor[start:end]
        norms.append(torch.norm(seg, p=2))
    norms_tensor = torch.stack(norms)
    prof = norms_tensor / (torch.sum(norms_tensor) + 1e-12)
    return prof


def quantize_int8_downlink(delta_w, slices):
    delta_w = torch.nan_to_num(delta_w, nan=0.0, posinf=0.0, neginf=0.0)
    recon_chunks = []
    for (start, end) in slices:
        seg = delta_w[start:end]
        s_min, s_max = seg.min().item(), seg.max().item()
        diff = s_max - s_min
        if diff > 1e-7:
            scale = diff / 255.0
            val = -s_min / scale
            zp = 0 if (math.isnan(val) or math.isinf(val)) else max(-128, min(127, round(val) - 128))
            q = torch.clamp(torch.round(seg / scale) + zp, -128, 127).to(torch.int8)
            deq = (q.float() - zp) * scale
        else:
            deq = seg.clone()
        recon_chunks.append(deq)
    return torch.cat(recon_chunks)


def evaluate_proxy_accuracy(model, proxy_loader):
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


# Pillar IV Activation Pruning with Explicit Round Logging
def execute_pillar4(model, proxy_train_loader, proxy_val_loader, prune_frac=0.15, do_finetune=False, arm_name="Unknown", round_idx=0):
    state_backup = copy.deepcopy(model.state_dict())

    # 1. Baseline Proxy Accuracy
    acc_clean_pre = evaluate_proxy_accuracy(model, proxy_val_loader)

    # 2. Prune dormant units on fc1
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
        k_prune_units = max(1, int(64 * prune_frac))
        dormant_units = torch.argsort(unit_means)[:k_prune_units]

        model.fc1.weight.data[dormant_units, :] = 0.0
        if model.fc1.bias is not None:
            model.fc1.bias.data[dormant_units] = 0.0
        pruned_units_str = str(dormant_units.tolist())

    # 3. Optional Fine-Tuning
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

    # 4. Post-Pruning Validation
    acc_clean_post = evaluate_proxy_accuracy(model, proxy_val_loader)

    # 5. Rollback Guard (threshold: clean drop > 2.0 percentage points)
    rolled_back = False
    if acc_clean_post < (acc_clean_pre - 2.0):
        model.load_state_dict(state_backup)
        rolled_back = True

    # 6. Explicit Round-Logged Telemetry to rollback_stage8.log
    log_line = (f"[{arm_name}] Round={round_idx} | Pillar4 | FT={do_finetune} | Prune={prune_frac:.2f} | "
                f"Units={pruned_units_str} | HeldOut Clean {acc_clean_pre:.2f}->{acc_clean_post:.2f} | "
                f"Rollback={rolled_back}\n")
    print(f"  [TELEMETRY] {log_line.strip()}")
    with open("rollback_stage8.log", "a") as rf:
        rf.write(log_line)


# Local Client Training
def train_local_client(client_id, global_weights, dataset, client_indices,
                       attack_type='none', seed=0, clip_c=5.0, batch_clip=5.0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = EdgeCNN().to(device)
    assign_params(model, global_weights)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=LR_LOCAL, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    subset_indices = client_indices[client_id]
    client_dataset = Subset(dataset, subset_indices)
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
    update = local_weights - global_weights

    # Byzantine Sign-Flip Attack (scale by -3.0)
    if attack_type == 'sign_flip':
        update = -3.0 * update

    update = torch.nan_to_num(update, nan=0.0, posinf=1.0, neginf=-1.0)
    u_norm = torch.norm(update, p=2)
    g_clipped = update * min(1.0, clip_c / (u_norm.item() + 1e-12))
    return torch.nan_to_num(g_clipped, nan=0.0, posinf=1.0, neginf=-1.0), u_norm.item()


# Server Aggregator
class EdgeShieldServer:
    def __init__(self, d, slices, ema_reference=False):
        self.d = d
        self.layer_slices = slices
        self.ema_reference = ema_reference
        self.prev_delta_global = None
        self.ema_delta_global = None
        self.cusum_states = None

    def aggregate_round(self, round_idx, client_uploads, defense_mode='edgeshield'):
        N = len(client_uploads)

        # Mode 1: Plain FedAvg (Uniform cleartext averaging)
        if defense_mode == 'fedavg':
            stacked = torch.stack([u['g_clipped'] for u in client_uploads])
            delta_w = torch.mean(stacked, dim=0)
            self.prev_delta_global = delta_w.clone()
            return delta_w, list(range(N)), np.ones(N) / N, np.zeros(N), [False] * N

        # Mode 2: EdgeShield-FL Dual-Channel Aggregator
        sketches = [torch.nan_to_num(u['s_hat']).cpu().numpy() for u in client_uploads]
        norms = [float(np.nan_to_num(u['M_hat'], nan=0.0)) for u in client_uploads]

        S = np.nan_to_num(np.array(sketches))
        S_norm = np.nan_to_num(S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12))
        Cm = np.nan_to_num(S_norm @ S_norm.T)

        # Tightness Tracker for Collusion (pairwise cosine > 0.72)
        TAU_TIGHT = 0.72
        colluding_mask = np.zeros(N, dtype=bool)
        for i in range(N):
            for j in range(i + 1, N):
                if Cm[i, j] > TAU_TIGHT:
                    colluding_mask[i] = True
                    colluding_mask[j] = True
        colluding = colluding_mask

        # CUSUM State Maintenance
        if self.cusum_states is None:
            self.cusum_states = np.zeros(N)

        if self.prev_delta_global is not None:
            if self.ema_delta_global is None:
                self.ema_delta_global = self.prev_delta_global.clone().float()
            else:
                alpha = 1.0 / 20.0 if self.ema_reference else 1.0
                self.ema_delta_global = (1 - alpha) * self.ema_delta_global + alpha * self.prev_delta_global.clone().float()

        ref_delta = self.ema_delta_global if self.ema_reference else self.prev_delta_global
        if ref_delta is not None and round_idx >= 5:
            s_prev = streaming_rademacher_projection(ref_delta, m=M_SKETCH, seed=round_idx * 100).cpu().numpy()
            s_prev = np.nan_to_num(s_prev)
            s_prev_norm = s_prev / (np.linalg.norm(s_prev) + 1e-12)
            rho_c = np.nan_to_num(-np.array([s_norm @ s_prev_norm for s_norm in S_norm]))

            CUSUM_NU = 0.01
            CUSUM_SIGMA = 0.1474
            drift = (rho_c - CUSUM_NU) / CUSUM_SIGMA
            self.cusum_states = np.maximum(0.0, self.cusum_states + drift)

        # Weight Redistribution
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
            weights[cap] = max(0.0, weight_remaining) / num_cap if weight_remaining >= 0 else 1.0 / floored.sum()

        admitted = [i for i in range(N) if weights[i] > 0]

        # Ring Quantization & Downlink INT8 Compression
        non_colluding_norms = [norms[i] for i in range(N) if not colluding[i]]
        M_norm = max(1e-6, float(np.median(non_colluding_norms))) if len(non_colluding_norms) > 0 else 1e-6
        num_adm = len(admitted)
        w_cap = (W_CAP_MULT / num_cap) if num_cap > 0 else (1.0 / max(1, num_adm))
        Delta_global = max(1e-9, (w_cap * M_norm) / R_MAX)

        weighted_updates = []
        for i in admitted:
            g = torch.nan_to_num(client_uploads[i]['g_clipped'])
            g_norm = torch.norm(g, p=2).item()
            g_capped = g * min(1.0, M_norm / (g_norm + 1e-12))
            u_i = weights[i] * g_capped
            u_scaled = u_i / (Delta_global + 1e-12)
            u_round = torch.floor(u_scaled + torch.rand_like(u_scaled))
            u_clamped = torch.clamp(u_round, -R_MAX, R_MAX)
            weighted_updates.append(u_clamped)

        recon_sum = torch.nan_to_num(torch.stack(weighted_updates).sum(dim=0)) if num_adm > 0 else torch.zeros(self.d, device=device)
        delta_w_clean = torch.nan_to_num(recon_sum * Delta_global)

        sigma_coord = (0.5 * M_norm) / math.sqrt(max(1, num_adm) * self.d)
        dilution = torch.randn_like(delta_w_clean) * sigma_coord
        delta_w_global = torch.nan_to_num(delta_w_clean + dilution)

        delta_w_quantized = quantize_int8_downlink(delta_w_global, self.layer_slices)
        self.prev_delta_global = delta_w_quantized.clone()
        return delta_w_quantized, admitted, weights, self.cusum_states.copy(), colluding.tolist()


# Model Evaluation on Test Sets
def evaluate_model_dual(model, full_loader, disjoint_loader, backdoor_loader):
    model.eval()

    # 1. Full 10,000-image test set
    corr_full, tot_full = 0, 0
    with torch.no_grad():
        for d, t in full_loader:
            d, t = d.to(device), t.to(device)
            pred = model(d).argmax(dim=1)
            corr_full += (pred == t).sum().item()
            tot_full += t.size(0)
    acc_full = (corr_full / tot_full) * 100.0 if tot_full > 0 else 0.0

    # 2. Disjoint 9,500-image test set (indices 500..9999)
    corr_disj, tot_disj = 0, 0
    with torch.no_grad():
        for d, t in disjoint_loader:
            d, t = d.to(device), t.to(device)
            pred = model(d).argmax(dim=1)
            corr_disj += (pred == t).sum().item()
            tot_disj += t.size(0)
    acc_disj = (corr_disj / tot_disj) * 100.0 if tot_disj > 0 else 0.0

    # 3. Backdoor ASR (continuity)
    corr_bd, tot_bd = 0, 0
    with torch.no_grad():
        for d, t in backdoor_loader:
            d, t = d.to(device), t.to(device)
            pred = model(d).argmax(dim=1)
            corr_bd += (pred == t).sum().item()
            tot_bd += t.size(0)
    asr_bd = (corr_bd / tot_bd) * 100.0 if tot_bd > 0 else 0.0

    return acc_full, acc_disj, asr_bd


# Simulation Runner for a Single Arm
def run_arm(arm_name, rounds=100, beta=0.20, defense_mode='edgeshield',
            attack_type='sign_flip', seed=42, enable_pruning=False,
            k_prune=25, ft_on=False, ema_reference=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*75}")
    print(f"LAUNCHING ARM: [{arm_name}]")
    print(f"CONFIG: Defense={defense_mode} | Attack={attack_type} | Rounds={rounds} | Seed={seed}")
    print(f"PRUNING: Enabled={enable_pruning} | K_prune={k_prune} | FT={ft_on} | TrainProxy=True")
    print(f"{'='*75}")

    # Partition training pool (indices 500..49999)
    client_indices = partition_dirichlet_client_pool(trainset, client_pool_indices, num_clients=NUM_CLIENTS, alpha=1.0, seed=seed)
    proxy_train_loader = DataLoader(proxy_train_subset, batch_size=64, shuffle=False)
    proxy_val_loader = DataLoader(proxy_val_subset, batch_size=64, shuffle=False)

    global_model = EdgeCNN().to(device)
    server = EdgeShieldServer(d=d_params, slices=layer_slices, ema_reference=ema_reference)

    num_att = int(round(beta * NUM_CLIENTS)) if attack_type != 'none' else 0
    byz_clients = set(range(NUM_CLIENTS - num_att, NUM_CLIENTS))

    history = {
        'round': [],
        'clean_acc_full': [],
        'clean_acc_disjoint': [],
        'backdoor_asr': [],
        'attacker_flooring_r30_60': None,
        'friendly_fire_r1_100': None,
        'admitted_history': []
    }

    init_full, init_disj, init_asr = evaluate_model_dual(global_model, full_test_loader, disjoint_test_loader, poisoned_test_loader)
    print(f"Round 0: Clean Full={init_full:.2f}%, Clean Disjoint={init_disj:.2f}%")
    history['round'].append(0)
    history['clean_acc_full'].append(init_full)
    history['clean_acc_disjoint'].append(init_disj)
    history['backdoor_asr'].append(init_asr)

    flooring_opps = 0
    flooring_hits = 0
    ff_opps = 0
    ff_hits = 0

    pbar = tqdm(range(1, rounds + 1), desc=f"Arm: {arm_name[:24]}")
    for t in pbar:
        global_weights = flatten_params(global_model)
        client_uploads = []

        for i in range(NUM_CLIENTS):
            is_byz = (i in byz_clients)
            client_attack = attack_type if is_byz else 'none'
            c_seed = seed * 100000 + t * 1000 + i

            g_i, unclipped_norm = train_local_client(
                client_id=i,
                global_weights=global_weights,
                dataset=trainset,
                client_indices=client_indices,
                attack_type=client_attack,
                seed=c_seed,
                clip_c=CLIP_C,
                batch_clip=BATCH_CLIP_NORM
            )

            # Obfuscated sketch Phase 1
            c_gen = torch.Generator(device=g_i.device)
            c_gen.manual_seed(c_seed)
            s_i = streaming_rademacher_projection(g_i, m=M_SKETCH, seed=t * 100)
            s_hat_i = s_i + torch.randn(s_i.shape, generator=c_gen, device=g_i.device) * SIGMA_S
            prof_i = compute_layer_profile(g_i, layer_slices)
            prof_hat_i = prof_i + torch.randn(prof_i.shape, generator=c_gen, device=g_i.device) * SIGMA_PROF
            lap_noise = np.random.default_rng(c_seed).laplace(0, B_M_LAPLACE)
            M_hat_i = max(0.0, float(torch.norm(g_i, p=2).item() + lap_noise))

            client_uploads.append({
                'g_clipped': g_i,
                's_hat': s_hat_i,
                'prof_hat': prof_hat_i,
                'M_hat': M_hat_i
            })

        # Server Aggregation
        delta_w_global, admitted, weights, cusum_st, colluding_list = server.aggregate_round(
            round_idx=t,
            client_uploads=client_uploads,
            defense_mode=defense_mode
        )

        # Apply update
        new_weights = global_weights + delta_w_global
        assign_params(global_model, new_weights)

        # Pillar IV Pruning (selected and gated on clean trainset proxy)
        if enable_pruning and (k_prune is not None) and (t % k_prune == 0):
            execute_pillar4(global_model, proxy_train_loader, proxy_val_loader,
                           prune_frac=0.15, do_finetune=ft_on, arm_name=arm_name, round_idx=t)

        # Track mechanism metrics
        if 30 <= t <= 60 and num_att > 0:
            for att_idx in byz_clients:
                flooring_opps += 1
                if weights[att_idx] <= 0.0101:
                    flooring_hits += 1

        if 1 <= t <= 100:
            honest_clients = [idx for idx in range(NUM_CLIENTS) if idx not in byz_clients]
            for h_idx in honest_clients:
                ff_opps += 1
                if weights[h_idx] <= 0.0101:
                    ff_hits += 1

        # Periodic model evaluation
        if t % 10 == 0 or t == rounds:
            acc_f, acc_d, asr = evaluate_model_dual(global_model, full_test_loader, disjoint_test_loader, poisoned_test_loader)
            history['round'].append(t)
            history['clean_acc_full'].append(acc_f)
            history['clean_acc_disjoint'].append(acc_d)
            history['backdoor_asr'].append(asr)
            pbar.set_postfix({'FullAcc': f"{acc_f:.2f}%", 'DisjAcc': f"{acc_d:.2f}%"})

    history['attacker_flooring_r30_60'] = (flooring_hits / flooring_opps * 100.0) if flooring_opps > 0 else 0.0
    history['friendly_fire_r1_100'] = (ff_hits / ff_opps * 100.0) if ff_opps > 0 else 0.0

    final_full = history['clean_acc_full'][-1]
    final_disj = history['clean_acc_disjoint'][-1]
    print(f"\nARM COMPLETE: [{arm_name}] -> Terminal Acc (Full): {final_full:.2f}%, (Disjoint): {final_disj:.2f}%")
    print(f"Mechanism: Attacker Flooring (30-60): {history['attacker_flooring_r30_60']:.2f}%, Friendly Fire: {history['friendly_fire_r1_100']:.2f}%")

    # Save terminal checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/stage8_{arm_name}_r100.pt"
    torch.save(global_model.state_dict(), ckpt_path)
    print(f"Saved terminal model checkpoint to: {ckpt_path}")

    return {
        'arm_name': arm_name,
        'final_clean_acc_full': final_full,
        'final_clean_acc_disjoint': final_disj,
        'final_backdoor_asr': history['backdoor_asr'][-1],
        'attacker_flooring_r30_60': history['attacker_flooring_r30_60'],
        'friendly_fire_r1_100': history['friendly_fire_r1_100'],
        'history': history
    }


# Experiment Suite Definition & Atomic Resumption Loop
RESULTS_FILE = "stage8_2x2_ablation_results.json"


def load_existing_results():
    # Auto-discover from /kaggle/input if running in Kaggle and file not in current directory
    if not os.path.exists(RESULTS_FILE) and os.path.exists("/kaggle/input"):
        import shutil
        for root, dirs, files in os.walk("/kaggle/input"):
            if RESULTS_FILE in files:
                src_path = os.path.join(root, RESULTS_FILE)
                print(f"Auto-discovered existing results file at: {src_path}")
                shutil.copy(src_path, RESULTS_FILE)
                break
        for root, dirs, files in os.walk("/kaggle/input"):
            if os.path.basename(root) == "checkpoints" and len(files) > 0:
                print(f"Auto-discovered checkpoints directory at: {root}")
                shutil.copytree(root, "checkpoints", dirs_exist_ok=True)
                break

    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, 'r') as f:
                data = json.load(f)
                print(f"Loaded existing results file: {len(data)} completed arms found.")
                return data
        except Exception as e:
            print(f"Warning: could not parse {RESULTS_FILE}: {e}")
    return {}


def save_atomic_results(results_dict):
    tmp_file = f"{RESULTS_FILE}.tmp"
    with open(tmp_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    os.replace(tmp_file, RESULTS_FILE)
    print(f"Atomically committed {len(results_dict)} arms to {RESULTS_FILE}")


def main():
    SEEDS = [42, 43, 44, 45, 46, 47]
    existing_results = load_existing_results()

    # Define the 12 Core Ablation Arms (Milestone 1)
    planned_arms = []

    # Factorial Cell (1, 0): EdgeShield Sketch/CUSUM ONLY (Pruning OFF: k_prune=999)
    for s in SEEDS:
        planned_arms.append({
            'arm_name': f"EdgeShield_NoPrune_SignFlip20_a1.0_s{s}",
            'defense_mode': 'edgeshield',
            'attack_type': 'sign_flip',
            'seed': s,
            'enable_pruning': False,
            'k_prune': 999,
            'ft_on': False,
            'cell': '(1,0) Sketch-Only'
        })

    # Factorial Cell (0, 1): FedAvg + Proxy Pruning ONLY (Uniform FedAvg + Pillar IV on Clean Train Proxy)
    for s in SEEDS:
        planned_arms.append({
            'arm_name': f"FedAvg_PruneOnly_SignFlip20_a1.0_s{s}",
            'defense_mode': 'fedavg',
            'attack_type': 'sign_flip',
            'seed': s,
            'enable_pruning': True,
            'k_prune': 25,
            'ft_on': False,
            'cell': '(0,1) Prune-Only'
        })

    # Optional Milestone 2: Factorial Cell (1, 1) EdgeShield Full Pipeline with Clean Train Proxy
    RUN_MILESTONE_2 = os.environ.get("RUN_MILESTONE_2", "1") == "1" and ("--milestone1_only" not in sys.argv)
    if RUN_MILESTONE_2:
        for s in SEEDS:
            planned_arms.append({
                'arm_name': f"EdgeShield_TrainProxy_SignFlip20_a1.0_s{s}",
                'defense_mode': 'edgeshield',
                'attack_type': 'sign_flip',
                'seed': s,
                'enable_pruning': True,
                'k_prune': 25,
                'ft_on': False,
                'cell': '(1,1) Full-Pipeline Clean-Proxy'
            })
    else:
        print("Note: Running Milestone 1 ONLY (12 core ablation arms). Milestone 2 excluded.")

    print(f"\nTotal planned arms in suite: {len(planned_arms)}")
    pending_arms = [a for a in planned_arms if a['arm_name'] not in existing_results]
    print(f"Already completed: {len(existing_results)} arms. Pending to run: {len(pending_arms)} arms.")

    for arm_cfg in pending_arms:
        arm_name = arm_cfg['arm_name']
        print(f"\n>>> Starting arm: {arm_name} [{arm_cfg['cell']}]")
        res = run_arm(
            arm_name=arm_name,
            rounds=100,
            beta=0.20,
            defense_mode=arm_cfg['defense_mode'],
            attack_type=arm_cfg['attack_type'],
            seed=arm_cfg['seed'],
            enable_pruning=arm_cfg['enable_pruning'],
            k_prune=arm_cfg['k_prune'],
            ft_on=arm_cfg['ft_on'],
            ema_reference=True
        )
        existing_results[arm_name] = res
        save_atomic_results(existing_results)

    # Statistical Evaluation & Decision Rules
    print("\n" + "=" * 75)
    print("STAGE 8: 2x2 FACTORIAL ABLATION STATISTICAL REPORT")
    print("=" * 75)

    # Matched Baseline Anchors from Frozen Table 4B / Table 8
    # FedAvg-SF Accuracies: s42: 59.22, s43: 57.88, s44: 58.11, s45: 57.69, s46: 58.14, s47: 57.09
    baseline_sf = {
        42: 59.22,
        43: 57.88,
        44: 58.11,
        45: 57.69,
        46: 58.14,
        47: 57.09
    }

    # Analyze Cell (1, 0): Sketch-Only
    d_sketch = []
    print("\nCell (1, 0): EdgeShield Sketch-Only (Pruning OFF)")
    for s in SEEDS:
        name = f"EdgeShield_NoPrune_SignFlip20_a1.0_s{s}"
        if name in existing_results:
            acc = existing_results[name]['final_clean_acc_full']
            d = acc - baseline_sf[s]
            d_sketch.append(d)
            print(f"  Seed {s}: Acc = {acc:.2f}%, SF = {baseline_sf[s]:.2f}%, D = {d:+.2f}%")

    # Analyze Cell (0, 1): Prune-Only
    d_prune = []
    print("\nCell (0, 1): FedAvg + Pruning-Only (Sketch OFF)")
    for s in SEEDS:
        name = f"FedAvg_PruneOnly_SignFlip20_a1.0_s{s}"
        if name in existing_results:
            acc = existing_results[name]['final_clean_acc_full']
            d = acc - baseline_sf[s]
            d_prune.append(d)
            print(f"  Seed {s}: Acc = {acc:.2f}%, SF = {baseline_sf[s]:.2f}%, D = {d:+.2f}%")

    def calc_ci(arr):
        if len(arr) < 2: return 0.0, 0.0, (0.0, 0.0), 1.0
        n = len(arr)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))
        se = std / math.sqrt(n)
        from scipy import stats
        t_crit = stats.t.ppf(0.975, df=n-1)
        ci = (mean - t_crit * se, mean + t_crit * se)
        t_stat = mean / se if se > 0 else 0.0
        p_val = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n-1))
        return mean, se, ci, p_val

    if len(d_sketch) == len(SEEDS) and len(d_prune) == len(SEEDS):
        try:
            m_sk, se_sk, ci_sk, p_sk = calc_ci(d_sketch)
            m_pr, se_pr, ci_pr, p_pr = calc_ci(d_prune)

            print("\n" + "=" * 75)
            print(f"Sketch-Only Recovery (D_sketch): mean = {m_sk:+.2f}%, SE = {se_sk:.2f}%, 95% CI: [{ci_sk[0]:.2f}%, {ci_sk[1]:.2f}%], p = {p_sk:.4f}")
            print(f"Prune-Only Recovery  (D_prune):  mean = {m_pr:+.2f}%, SE = {se_pr:.2f}%, 95% CI: [{ci_pr[0]:.2f}%, {ci_pr[1]:.2f}%], p = {p_pr:.4f}")
            print("=" * 75)

            # Pre-Registered Decision Branches
            if ci_sk[0] > 0.0 and m_sk >= 2.0 and m_pr <= 1.0:
                print("DECISION: BRANCH 1 CONFIRMED -> Sketch/CUSUM tracking is the primary causal driver.")
            elif ci_pr[0] > 0.0 and m_pr >= 2.0 and m_sk <= 1.0:
                print("DECISION: BRANCH 2 CONFIRMED -> Server proxy pruning is the primary causal driver.")
            elif m_sk > 0.0 and m_pr > 0.0:
                print("DECISION: BRANCH 3 CONFIRMED -> Synergistic interaction between sketch tracking and pruning.")
            else:
                print("DECISION: BRANCH 4 -> Underpowered / SGD noise domination. Extend to seeds 48-53.")
        except Exception as e:
            print(f"Could not compute CI (scipy might not be installed): {e}")

    print("\nStage 8 Suite Execution Finished.")


if __name__ == '__main__':
    main()
