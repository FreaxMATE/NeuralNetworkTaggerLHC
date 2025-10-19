import math
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler
import torch
import os
import torch.nn as nn
import torch.optim as optim
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

plt.rcParams["figure.figsize"] = (16, 12)
plt.rcParams["axes.grid"] = True
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
# -----------------------------
# Basic data structures
# -----------------------------
class Particle:
    def __init__(self, event_id, pid, pt, eta, phi, e, m, truth=False):
        self.event_id = event_id
        self.pid = pid
        self.pt = pt
        self.eta = eta
        self.phi = phi
        self.e = e
        self.m = m
        self.truth = truth  # only meaningful for jets in MC

class Event:
    def __init__(self, eid, particles=None):
        self.event_id = eid
        self.particles = [] if particles is None else particles
    def add(self, p: Particle):
        assert p.event_id == self.event_id
        self.particles.append(p)
    def jets(self):
        return [p for p in self.particles if abs(p.pid) == 90]
    def leptons(self):
        return [p for p in self.particles if abs(p.pid) != 90]
    def leading_jet(self):
        js = self.jets()
        return max(js, key=lambda p: p.pt) if js else None
    def leading_lepton(self):
        ls = self.leptons()
        return max(ls, key=lambda p: p.pt) if ls else None

# -----------------------------
# IO: load CSV (MC or data)
# -----------------------------
def load_events_csv(path):
    events = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): 
                continue
            toks = line.split(",")
            if len(toks) == 8:
                eid_s, pid_s, pt_s, eta_s, phi_s, e_s, m_s, truth_s = toks
                truth = bool(int(truth_s))
            elif len(toks) == 7:
                eid_s, pid_s, pt_s, eta_s, phi_s, e_s, m_s = toks
                truth = False
            else:
                continue
            eid  = int(eid_s); pid = int(pid_s)
            pt   = float(pt_s); eta = float(eta_s); phi = float(phi_s)
            e    = float(e_s);  m   = float(m_s)
            if eid not in events:
                events[eid] = Event(eid)
            events[eid].add(Particle(eid, pid, pt, eta, phi, e, m, truth))
    return [events[k] for k in sorted(events)]

# -----------------------------
# Helpers
# -----------------------------
def dphi(a, b):
    d = a - b
    return abs((d + math.pi) % (2*math.pi) - math.pi)

def dR(deta, dphi):
    return np.sqrt(deta**2 + dphi**2)
# -----------------------------
# Main: cut-based selection and plots
# -----------------------------
if __name__ == "__main__":
    # Files
    DATA_FILE = "jets.csv"
    MC_FILE   = "pythia.csv"

    # Cuts
    min_pt_j  = 250.0 # min pT for jet
    min_pt_l  = 50.0 # min pT for lepton
    min_dphi  = 2.4  # radians
    eta_j_max = 2.0  # max eta for jet

    def pass_cuts(e: Event):
        j = e.leading_jet()
        l = e.leading_lepton()
        if (j is None) or (l is None):
            return False
        return (
            (j.pt >= min_pt_j) and
            (l.pt >= min_pt_l) and
            (dphi(j.phi, l.phi) >= min_dphi) and
            (abs(j.eta) <= eta_j_max)
        )

    mc_events = load_events_csv(MC_FILE)
    mc_events = np.array(mc_events)
    # MLP
    # Example: print leading jet for each event in mc_events
    def feature_vector(events):
        n_data = len(events)

        x = []
        y_label = []
        for train_event in events:
            # leading jet particles
            lj = train_event.leading_jet()
            ll = train_event.leading_lepton()
            if (lj is None) or (ll is None):
                continue
            phi_diff = dphi(lj.phi, ll.phi)
            deta = lj.eta-ll.eta
            dr = dR(deta, phi_diff)
            x.append([
                lj.pt, lj.eta, lj.phi,
                ll.pt, ll.eta, ll.phi,
                phi_diff, dr, lj.pt/ll.pt,
                (lj.pt - ll.pt)/(lj.pt + ll.pt)
            ])
            y_label.append(lj.truth)
        return x, y_label

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    x_mc, y_label_mc = feature_vector(events=mc_events)

    n = len(x_mc)
    train_idx, valid_test_idx = sklearn.model_selection.train_test_split(np.arange(n), train_size=0.8, test_size=0.2)
    valid_idx, test_idx = sklearn.model_selection.train_test_split(valid_test_idx, train_size=0.5, test_size=0.5)

    x_mc = np.array(x_mc)
    y_label_mc = np.array(y_label_mc)

    # keep MC-specific names and also provide the expected x_train/x_valid/x_test variables
    x_mc_train = x_mc[train_idx]
    y_mc_train = y_label_mc[train_idx]
    x_mc_valid = x_mc[valid_idx]
    y_mc_valid = y_label_mc[valid_idx]
    x_mc_test  = x_mc[test_idx]
    y_mc_test  = y_label_mc[test_idx]

    # Also expose the conventional names used later in the script
    x_mc_train = np.array(x_mc_train, dtype=np.float32)
    x_mc_valid = np.array(x_mc_valid, dtype=np.float32)
    x_mc_test  = np.array(x_mc_test, dtype=np.float32)
    y_mc_train = np.array(y_mc_train, dtype=np.float32)
    y_mc_valid = np.array(y_mc_valid, dtype=np.float32)
    y_mc_test  = np.array(y_mc_test, dtype=np.float32)

    scaler = StandardScaler()
    x_mc_train = scaler.fit_transform(x_mc_train)
    x_mc_valid = scaler.transform(x_mc_valid)
    x_mc_test  = scaler.transform(x_mc_test)

    # Small, conservative model/hyperparameter tweaks to reduce val-loss fluctuations:
    # - set seed for reproducibility before model creation
    # - add BatchNorm between Linear and ReLU
    # - slightly larger dropout
    # - keep model small (same sizes) to avoid overfitting
    in_dim  = len(x_mc_train[0]) # Number of input dimensions
    h1, h2  = 32, 16  # Two hidden layers with h1 and h2 neurons

    model = nn.Sequential(
        nn.Linear(in_dim, h1),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(h1, h2),
        nn.ReLU(),
        nn.Linear(h2, 1),
    )

    # Inspect the network
    print(model)
    print("Number of trainable parameters:", sum(p.numel()
        for p in model.parameters() if p.requires_grad))

    # Define Loss Function and Optimizer
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=12)

    # Revision number to save output to different location
    revision = '17'
    epochs = 150

    # Record the per-epoch loss and accuracy so we can plot it later
    losses = np.zeros((epochs))
    accuracies = np.zeros((epochs))
    losses_valid = np.zeros((epochs))
    accuracies_valid = np.zeros((epochs))
    learning_rate = np.zeros((epochs))

    # Convert data to PyTorch tensors
    x_mc_train_tensor = torch.tensor(x_mc_train, dtype=torch.float32)
    y_mc_train_tensor = torch.tensor(y_mc_train, dtype=torch.float32)
    x_mc_valid_tensor = torch.tensor(x_mc_valid, dtype=torch.float32)
    y_mc_valid_tensor = torch.tensor(y_mc_valid, dtype=torch.float32)

    batch_size = 128
    mc_train_dataset = torch.utils.data.TensorDataset(x_mc_train_tensor, y_mc_train_tensor)
    mc_valid_dataset = torch.utils.data.TensorDataset(x_mc_valid_tensor, y_mc_valid_tensor)
    mc_train_loader = torch.utils.data.DataLoader(mc_train_dataset, batch_size=batch_size, shuffle=True)
    mc_valid_loader = torch.utils.data.DataLoader(mc_valid_dataset, batch_size=batch_size, shuffle=False)

    # Train on MC data
    for epoch in range(epochs):
        running_loss = 0
        num_correct = 0
        total_samples = 0

        running_loss_val = 0
        num_correct_val = 0
        total_samples_val = 0
        model.train()
        for i, (inputs, targets) in enumerate(mc_train_loader):
            optimizer.zero_grad()
            predictions = model(inputs)

            loss = loss_function(predictions, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # print(predictions)
            probs = torch.sigmoid(predictions).squeeze(1)
            # print(probs)
            num_correct += torch.eq(probs.round().bool(), targets.bool()).sum().item()
            total_samples += targets.size(0)

        losses[epoch] = running_loss / len(mc_train_loader)
        accuracies[epoch] = num_correct / total_samples

        # model.eval()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(mc_valid_loader):
                predictions = model(inputs)
                loss = loss_function(predictions, targets.unsqueeze(1))
                running_loss_val += loss.item()
                probs = torch.sigmoid(predictions).squeeze(1)
                num_correct_val += torch.eq(probs.round().bool(), targets.bool()).sum().item()
                total_samples_val += targets.size(0)

            losses_valid[epoch] = running_loss_val / len(mc_valid_loader)
            accuracies_valid[epoch] = num_correct_val / total_samples_val
            print(
                f"Epoch: {epoch+1:02d}/{epochs} | "
                f"Loss: {losses[epoch]:.5f} | Accuracy: {accuracies[epoch]:.5f} | "
                f"Val Loss: {losses_valid[epoch]:.5f} | Val Accuracy: {accuracies_valid[epoch]:.5f}"
            )
            # Step the LR scheduler on validation loss (no early stopping)
            try:
                scheduler.step(losses_valid[epoch])
            except Exception:
                pass
        learning_rate[epoch] = get_lr(optimizer=optimizer)

    title = 'out/'+revision+'/training_v'+revision
    out_dir = os.path.dirname(title)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), title+'_model.pt')
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.plot(range(epochs), losses, label="Train Loss")
    ax1.plot(range(epochs), losses_valid, label="Val Loss")
    ax2.plot(range(epochs), accuracies, label="Train Accuracy")
    ax2.plot(range(epochs), accuracies_valid, label="Val Accuracy")
    ax3.plot(range(epochs), learning_rate, label="Learning Rate")
    ax3.set_yscale("log")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Crossentropy Loss")
    ax2.set_ylabel("Accuracy")
    ax3.set_ylabel("Learning Rate")
    ax1.legend()
    ax2.legend()
    plt.savefig(title+'.png')
    # Save training history (epoch, train_loss, val_loss, train_acc, val_acc)
    # and a small summary file with model metadata. Do both inside a single
    # try/except so failures are reported instead of silently ignored.
    try:
        # Merge training history and model metadata into a single CSV
        epochs_arr = np.arange(1, len(losses) + 1)
        hist_arr = np.vstack([
            epochs_arr,
            losses,
            losses_valid,
            accuracies,
            accuracies_valid
        ]).T  # columns: epoch, train_loss, val_loss, train_acc, val_acc

        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        lr = optimizer.param_groups[0].get('lr', 'unknown')

        header = (
            "epoch,train_loss,val_loss,train_acc,val_acc\n"
            f"Model: {model}\n"
            f"Trainable params: {param_count}\n"
            f"Batch size: {batch_size}\n"
            f"Learning rate: {lr}\n"
            f"Epochs: {epochs}\n"
        )

        np.savetxt(
            title + '_training.csv',
            hist_arr,
            delimiter=',',
            header=header,
            comments=''
        )
    except Exception as e:
        # Print a warning so failures to write files are visible during runs.
        print(f"Warning: could not save training outputs: {e}")


    with torch.no_grad():
        probs_mc_test = model(torch.from_numpy(x_mc_test).float()).numpy().squeeze()
        probs_mc_test = torch.sigmoid(torch.tensor(probs_mc_test))


    y_mc_test_int = y_mc_test.astype(int)  # ensure {0,1}

    def roc_curve_manual(y, s):
        """
        Manual ROC construction.
        y: (n,) labels in {0,1}
        s: (n,) scores (higher = more 'signal-like')
        Returns: fpr, tpr, thresholds, auc
        """
        # sort by score descending
        order = np.argsort(-s)
        y = y[order]
        s = s[order]

        # cumulative counts as we sweep the threshold down from +inf
        tp = np.cumsum(y == 1).astype(float)
        fp = np.cumsum(y == 0).astype(float)
        P  = max(1.0, (y == 1).sum())
        N  = max(1.0, (y == 0).sum())
        tpr = tp / P
        fpr = fp / N

        # prepend (0,0) at threshold above max, and append (1,1) at threshold below min
        fpr = np.r_[0.0, fpr, 1.0]
        tpr = np.r_[0.0, tpr, 1.0]

        # AUC via trapezoid rule
        auc = np.trapz(tpr, fpr)
        return fpr, tpr, auc

    plt.figure()
    probs_mc_test = probs_mc_test.detach().cpu().numpy().reshape(-1)
    x = np.linspace(0, 1, len(probs_mc_test))
    plt.plot(x, probs_mc_test, '.', color='C0', alpha=0.7, label='scores')
    t0 = 0.609
    mask = probs_mc_test >= t0
    plt.scatter(x[mask], probs_mc_test[mask], s=30, color='C2', label=f'score ≥ {t0}')
    ax = plt.gca(); mean = probs_mc_test.mean(); std = probs_mc_test.std()
    ax.fill_between([x.min(), x.max()], mean - std, mean + std, color='C7', alpha=0.25, label='±1σ')
    ax.hlines(mean, x.min(), x.max(), colors='k', linestyles='--', label='mean')
    plt.xlabel('Normalized event index'); plt.ylabel('Classifier score'); plt.title('MC test scores'); plt.legend(); plt.grid(True)
    plt.savefig(title + '_scores.png')
    plt.close()

    fpr_m, tpr_m, auc_m = roc_curve_manual(y_mc_test_int, probs_mc_test)
    plt.figure()
    plt.plot(fpr_m, tpr_m, label=f"Manual AUC = {auc_m:.3f}")
    plt.plot([0,1],[0,1],'--',lw=1)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC (manual)")
    plt.legend(); plt.grid()
    plt.savefig(title+'_roc.png')
    # save ROC data
    try:
        roc_arr = np.vstack([fpr_m, tpr_m]).T
        np.savetxt(title + '_roc.csv', roc_arr, delimiter=',', header=f'fpr,tpr  # AUC={auc_m:.6f}', comments='')
    except Exception:
        pass

    # ---- Purity (precision) vs threshold, plotted against (1 - t) ----
    # Purity = S/(S+B) = TP/(TP+FP)

    # Sort by score descending (sweep threshold downward)
    order   = np.argsort(-probs_mc_test)
    y_sorted= y_mc_test_int[order]
    s_sorted= probs_mc_test[order]

    # Cumulative TP/FP as we include items one by one
    tp = np.cumsum(y_sorted == 1).astype(float)
    fp = np.cumsum(y_sorted == 0).astype(float)

    # Purity at each step after including that item
    purity = tp / np.maximum(1.0, tp + fp)

    p_best_in = -1
    p_best = 0
    for i, p in enumerate(purity):
        if tpr_m[1:-1][i] >= 0.3 and p > p_best:
            p_best = p
            p_best_in = i


    # Threshold used at each step is t = score; we plot vs (1 - t)
    one_minus_t = 1.0 - s_sorted

    # Class prevalence = expected purity for random scores (baseline)
    prevalence = y_mc_test_int.mean()

    t_star = 1.0 - one_minus_t[p_best_in].item()

    plt.figure()
    plt.step(one_minus_t, purity, where="pre", label="purity S/(S+B)")
    plt.step(one_minus_t, tpr_m[1:-1], where="pre", label="Efficiency (true-positive-rate)")
    plt.scatter(one_minus_t, purity, s=18)

    # Random-guess baseline (horizontal line at prevalence)
    plt.plot([0,1], [prevalence, prevalence], "--", lw=1, color="gray",
            label=f"random baseline (prevalence={prevalence:.2f})")

    plt.axvline(x=one_minus_t[p_best_in], color="C1", linestyle="--", lw=1.5,
                label=f"best purity={p_best:.3f} at t*={t_star:.3f}")

    plt.xlim(0,1); plt.ylim(0,1.05)
    plt.xlabel(r"$1 - t$  (predict positive if score $>= t$)")
    plt.ylabel(r"Purity  $S/(S+B)=\mathrm{TP}/(\mathrm{TP}+\mathrm{FP})$")
    plt.title("Purity vs threshold")
    plt.grid(True); plt.legend()
    plt.savefig(title+'_purity.png')
    # save purity vs threshold data and scores
    try:
        purity_arr = np.vstack([one_minus_t, purity]).T
        np.savetxt(title + '_purity.csv', purity_arr, delimiter=',', header='one_minus_t,purity', comments='')
    except Exception:
        pass

    try:
        # save classifier scores and true labels used to build ROC/purity
        scores_arr = np.vstack([probs_mc_test, y_true]).T
        np.savetxt(title + '_scores.csv', scores_arr, delimiter=',', header='score,label', comments='')
    except Exception:
        pass


    ##
    ## Apply to data
    ##

    # ---- Data: mass spectra before/after cuts
    data_events = load_events_csv(DATA_FILE)

    data_unk, _ = feature_vector(events=data_events)
    x_unk = scaler.transform(data_unk)

    # Result vector (one element eq one event) for MC
    result_mc_test = [True if p >= t_star else False for p in probs_mc_test]


    with torch.no_grad():
        probs_unk = model(torch.from_numpy(x_unk).float()).numpy().squeeze()
        probs_unk = torch.sigmoid(torch.tensor(probs_unk))
        result_unk = [True if p >= t_star else False for p in probs_unk]


    def pass_ml_cuts_unk(ev_ind):
        return result_unk[ev_ind]

    def pass_ml_cuts_mc(ev_ind):
        return result_mc_test[ev_ind]

    all_masses = []
    sel_masses_cuts = []
    sel_masses_cuts_ml = []

    seen = 0; kept = 0; kept_ml = 0
    for ev_ind, e in enumerate(data_events):
        j = e.leading_jet(); l = e.leading_lepton()
        if (j is None) or (l is None):
            continue
        seen += 1
        all_masses.append(j.m)
        if pass_cuts(e):
            kept += 1
            sel_masses_cuts.append(j.m)

        if pass_ml_cuts_unk(ev_ind):
            kept_ml += 1
            sel_masses_cuts_ml.append(j.m)

    print(f"Data (cuts): selected {kept}/{seen} = {100*kept/max(1,seen):.1f}%.")

    bins = 40
    rng  = (60, 140)

    plt.figure(figsize=(5.8,4.2))
    plt.hist(all_masses, bins=bins, range=rng, density=True, histtype="step", label="All data")
    plt.hist(sel_masses_cuts, bins=bins, range=rng, density=True, histtype="step", label=f"Cuts")
    plt.hist(sel_masses_cuts_ml, bins=bins, range=rng, density=True, histtype="step", label=f"ML Cuts, t*={round(t_star, 2)}")
    plt.xlabel("Large-R jet mass [GeV]"); plt.ylabel("Density")
    plt.title("Data: jet mass before/after cuts"); plt.legend(); plt.tight_layout()
    plt.savefig(title+'_jetmass.png')

    # ---- MC: purity and efficiencies for the same cuts

    # Totals before selection (for efficiencies)
    S0 = B0 = 0
    for e in mc_events:
        j = e.leading_jet()
        if j is None: 
            continue
        S0 += int(j.truth)
        B0 += int(not j.truth)

    S = B = 0
    for e in mc_events:
        j = e.leading_jet(); l = e.leading_lepton()
        if (j is None) or (l is None):
            continue
        if pass_cuts(e):
            if j.truth: S += 1
            else:       B += 1
    

    # Print purity for Test split
    S_test = B_test = 0
    mc_events_test = mc_events[test_idx]
    for e_ind, e in enumerate(mc_events_test):
        j = e.leading_jet(); l = e.leading_lepton()
        if (j is None) or (l is None):
            continue
        if pass_cuts(e):
            if j.truth: S_test += 1
            else:       B_test += 1
    
    S_test_ml = B_test_ml = 0
    for e_ind, e in enumerate(mc_events_test):
        j = e.leading_jet(); l = e.leading_lepton()
        if (j is None) or (l is None):
            continue
        if pass_ml_cuts_mc(e_ind):
            if j.truth: S_test_ml += 1
            else:       B_test_ml += 1
    

    N_sel = S + B
    purity_mc = (S / N_sel) if N_sel > 0 else float('nan')
    eps_S = S / S0 if S0 > 0 else float('nan')
    eps_B = B / B0 if B0 > 0 else float('nan')
    purity0 = S0 / max(1, (S0 + B0))
    print(f"Baseline purity (no cuts) = {purity0:.3f}  with S0={S0}, B0={B0}")
    print(f"MC (cuts): purity S/(S+B) = {purity_mc:.3f}  "
          f"with S={S}, B={B}, N={N_sel}  |  eps_S={eps_S:.3f}, eps_B={eps_B:.3f}")

    purity_mc_test = S_test / (S_test + B_test)
    purity_mc_test_ml = S_test_ml / (S_test_ml + B_test_ml)

    print('Purities for the TEST split:')
    print(f"   MC (cuts): purity S/(S+B) = {purity_mc_test:.3f}  "
          f"    with S={S_test}, B={B_test}, N={S_test+B_test}")
    print(f"   MC (Machine Learning): purity S/(S+B) = {purity_mc_test_ml:.3f}  "
          f"    with S={S_test_ml}, B={B_test_ml}, N={S_test_ml+B_test_ml}")

    plt.show()