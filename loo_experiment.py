"""
Leave-one-out experiment: resting-state GEC + LR-weighted MDMA perturbation.

For each left-out subject i:
  Training subjects {0..10} \ {i}:
    1. Fit GEC_pla_j  via Pyfurcate_GEC  (targets FC_pla_j)
    2. Fit alpha_j    via Pyfurcate      (targets FC_mdma_j, offset=GEC_pla_j)
  Average alpha across training subjects -> alpha_mean
  Test on subject i:
    3. Fit GEC_pla_i  via Pyfurcate_GEC  (targets FC_pla_i)
    4. Predict GEC_mdma_i = GEC_pla_i + einsum(alpha_mean, basis) * mask
    5. Run Hopf forward, compare sim FC to FC_mdma_i
"""

import numpy as np
import scipy.io as sio
import pickle
import torch
import os

import Pyfurcate     as PF
import Pyfurcate_GEC as PF_GEC

# ── Data ──────────────────────────────────────────────────────────────────────
mat      = sio.loadmat('ROI_mdma_maas_timeseries_PSC_noGSR_scrub.mat')
ts_data  = mat['roi_timeseries'][0, 0]['schaefer232']  # (11, 2): col0=mdma, col1=pla
SC       = np.load('SC.npy')
Freqs    = np.load('Freqs.npy')

with open('lr_matrices.pkl', 'rb') as f:
    lr_dict  = pickle.load(f)
lr_names = sorted(lr_dict.keys())
n_lr     = len(lr_names)

N_SUBJECTS = ts_data.shape[0]  # 11
TR         = 1.4               # seconds
timings    = PF.fmri_params(TR, 256 * TR, verbose=True)

# SC mask
mask_np = np.where(SC > 0, 1.0, 0.0)
np.fill_diagonal(mask_np, 0.0)

# Pre-compute empirical FC for every subject × condition
print("Computing empirical FC matrices...")
FC_pla  = []
FC_mdma = []
for s in range(N_SUBJECTS):
    ts_m = ts_data[s, 0].astype(np.float64)   # MDMA  (256, 232)
    ts_p = ts_data[s, 1].astype(np.float64)   # PLA   (256, 232)
    FC_mdma.append(np.corrcoef(ts_m.T))
    FC_pla.append(np.corrcoef(ts_p.T))
print(f"  Done — {N_SUBJECTS} subjects, FC shape {FC_pla[0].shape}")

# ── Shared optimisation settings ──────────────────────────────────────────────
GEC_KWARGS = dict(
    C                  = SC,
    use_analytical_grad= False,
    optimizer_type     = 'adam',
    lr                 = 2e-3,
    min_lr             = 5e-5,
    grad_clip          = 1.0,
    weight_decay       = 1e-5,
    tbptt_frac         = 0.4,
    teq_opt            = 60.0,
    tmax_opt           = None,
    n_simulations      = 2,
    n_iterations       = 500,
    scheduler_patience = 80,
    max_gec            = 0.5,
    gec_init_max       = 0.1,
    allow_negative     = True,
    use_sc_mask        = True,
    use_filt           = False,
    verbose            = False,
    w                  = Freqs,
    device             = 'cpu',
    **timings,
)

# ── LOO loop ──────────────────────────────────────────────────────────────────
results = []

for left_out in range(N_SUBJECTS):
    print(f"\n{'='*60}")
    print(f"  LOO fold {left_out+1}/{N_SUBJECTS}  (subject {left_out} held out)")
    print(f"{'='*60}")

    train_idx = [s for s in range(N_SUBJECTS) if s != left_out]

    # ── Stage 1 & 2: train on each training subject ──────────────────────────
    alpha_list = []

    for j in train_idx:
        print(f"  Subject {j}: fitting resting GEC...", flush=True)
        res_pla = PF_GEC.optimize_GEC(FC_pla[j], **GEC_KWARGS)
        GEC_pla_j = res_pla['optimized_GEC']

        print(f"  Subject {j}: fitting LR perturbation (offset=GEC_pla)...", flush=True)
        res_mdma = PF.optimize_GEC(
            FC_mdma[j],
            C_offset  = GEC_pla_j,
            l1_lambda = 1e-4,
            **GEC_KWARGS,
        )
        alpha_list.append(res_mdma['optimized_alpha'])

        fc_end   = res_mdma['fc_history'][-1]
        loss_end = res_mdma['loss_history'][-1]
        print(f"    Subject {j} train — loss: {loss_end:.6f}  |  "
              f"FC corr: {fc_end:.4f}  |  "
              f"alpha range: [{res_mdma['optimized_alpha'].min():.3f}, "
              f"{res_mdma['optimized_alpha'].max():.3f}]")

    # ── Average alpha across training subjects ────────────────────────────────
    alpha_mean = np.mean(alpha_list, axis=0)   # (n_lr,)
    print(f"\n  alpha_mean range: [{alpha_mean.min():.4f}, {alpha_mean.max():.4f}]")

    # ── Stage 3: fit resting GEC for left-out subject ─────────────────────────
    print(f"  Subject {left_out}: fitting resting GEC (test)...", flush=True)
    res_pla_test = PF_GEC.optimize_GEC(FC_pla[left_out], **GEC_KWARGS)
    GEC_pla_test = res_pla_test['optimized_GEC']

    # ── Stage 4: predict MDMA GEC ─────────────────────────────────────────────
    basis_np = np.stack([lr_dict[k] for k in lr_names], axis=0).astype(np.float64)
    GEC_delta_pred = np.einsum('k,kij->ij', alpha_mean, basis_np) * mask_np
    GEC_mdma_pred  = GEC_pla_test + GEC_delta_pred

    # ── Stage 5: evaluate — run Hopf forward and compare to FC_mdma ───────────
    print(f"  Evaluating predicted MDMA FC...", flush=True)
    model = PF.HopfModel(
        nnodes   = SC.shape[0],
        C        = GEC_mdma_pred,
        backend  = 'numpy',
        G=1, a=0, w=Freqs, beta=1e-2,
        **timings,
    )
    sim_results, _ = model.forward()
    FC_pred = np.corrcoef(sim_results[:, :, 0].T)

    triu = np.triu_indices(SC.shape[0], k=1)
    r_pred    = float(np.corrcoef(FC_pred[triu], FC_mdma[left_out][triu])[0, 1])
    mse_pred  = float(np.mean((FC_pred[triu] - FC_mdma[left_out][triu]) ** 2))

    # Baseline: how well does GEC_pla alone predict FC_mdma?
    model_base = PF.HopfModel(
        nnodes   = SC.shape[0],
        C        = GEC_pla_test,
        backend  = 'numpy',
        G=1, a=0, w=Freqs, beta=1e-2,
        **timings,
    )
    sim_base, _ = model_base.forward()
    FC_base  = np.corrcoef(sim_base[:, :, 0].T)
    r_base   = float(np.corrcoef(FC_base[triu], FC_mdma[left_out][triu])[0, 1])
    mse_base = float(np.mean((FC_base[triu] - FC_mdma[left_out][triu]) ** 2))

    print(f"  Subject {left_out} test  — loss: {mse_pred:.6f}  |  FC corr: {r_pred:.4f}")
    print(f"  Subject {left_out} base  — loss: {mse_base:.6f}  |  FC corr: {r_base:.4f}")
    print(f"  Delta FC corr: {r_pred - r_base:+.4f}  |  Delta loss: {mse_pred - mse_base:+.6f}")

    results.append({
        'subject':       left_out,
        'r_predicted':   r_pred,
        'r_baseline':    r_base,
        'mse_predicted': mse_pred,
        'mse_baseline':  mse_base,
        'alpha_mean':    alpha_mean,
        'GEC_pla':       GEC_pla_test,
        'GEC_mdma_pred': GEC_mdma_pred,
    })

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  LOO SUMMARY")
print(f"{'='*60}")
r_pred_all    = [r['r_predicted']   for r in results]
r_base_all    = [r['r_baseline']    for r in results]
mse_pred_all  = [r['mse_predicted'] for r in results]
mse_base_all  = [r['mse_baseline']  for r in results]
print(f"  {'Subj':>5}  {'r_pred':>8}  {'r_base':>8}  {'Δr':>7}  "
      f"{'mse_pred':>10}  {'mse_base':>10}  {'Δmse':>10}")
for r in results:
    print(f"  {r['subject']:>5}  {r['r_predicted']:>8.4f}  {r['r_baseline']:>8.4f}  "
          f"{r['r_predicted']-r['r_baseline']:>+7.4f}  "
          f"{r['mse_predicted']:>10.6f}  {r['mse_baseline']:>10.6f}  "
          f"{r['mse_predicted']-r['mse_baseline']:>+10.6f}")
print(f"  {'mean':>5}  {np.mean(r_pred_all):>8.4f}  {np.mean(r_base_all):>8.4f}  "
      f"{np.mean(r_pred_all)-np.mean(r_base_all):>+7.4f}  "
      f"{np.mean(mse_pred_all):>10.6f}  {np.mean(mse_base_all):>10.6f}  "
      f"{np.mean(mse_pred_all)-np.mean(mse_base_all):>+10.6f}")

np.save('loo_results.npy', results)
print("\nSaved results to loo_results.npy")
