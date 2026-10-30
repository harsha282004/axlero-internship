"""Margin-sensitivity audit: original vs padding-augmented DenseNet121, seeds 42/43/44.
VALIDATION only, inference only; the test split is never read; no training; checkpoints untouched.

Why: the padding-augmented models resisted grey padding but showed new false positives when the outer 10%
margin was filled with the image mean. Hypotheses the condition set can discriminate (they are hypotheses,
not findings):
  H1 grey-frame: the model reads a grey/mean frame as "padding"  -> effect for grey128/imgmean fills, not black/trim
  H2 border-content: edge content (angles, shoulders, markers) matters -> effect for every fill AND trim
  H3 apparent scale/position: the visible region's size/aspect changes -> effect tied to trim / black merge

Conventions: original-image pixel coordinates (x right, y down), size w x h, BEFORE any resizing. All
perturbations act on the 8-bit grayscale original; then the checkpoint's saved pipeline follows unchanged:
Grayscale(3) -> LetterboxSquare(224, fill) -> ToTensor -> ImageNet Normalize.
  m_f   = round(f * min(w, h));  v = round(mean pixel value of the whole original image)
  margin{5,10}_{black,grey128,imgmean}: pixels with x < m_f or x >= w-m_f or y < m_f or y >= h-m_f set to the
        fill (0, 128 or v); padding bars stay black (fill 0)
  pad_imgmean                : letterbox bars filled with v, image untouched
  pad128_margin128_10        : margin (10%) AND letterbox bars filled with 128
  padmean_margin10mean       : margin (10%) AND letterbox bars filled with v
  trim{5,10}                 : crop to [m_f, w-m_f) x [m_f, h-m_f) (the SAME pixels the margin masks hide),
        size (w-2m_f) x (h-2m_f); then the saved pipeline: longer side scaled to 224 (bilinear, antialiased),
        centred on a black canvas. Content is magnified by about 1/(1-2f) and the aspect ratio shifts slightly
  ring10_interior            : band of width 0.10*min(w,h) between insets 0.20 and 0.30 of min(w,h) set to v
Black margin + black padding equals margin{f}_black, so no separate combined condition exists for black.
Thresholds are each checkpoint's ORIGINAL saved validation threshold (never retuned).

Run from project-1-healthcare/ (output dir must not already hold results):
    .venv\\Scripts\\python.exe scripts\\margin_sensitivity_audit.py --data-root <DATASET_ROOT>
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import functional as TF
from tqdm import tqdm

import size_matched_stress_test_all as allm                               # existing, unchanged
from gradcam import denorm                                                # existing, unchanged
from perturbation_audit_densenet import (GRAY3, load_gray, mask_margin, mask_ring,  # existing, unchanged
                                         metric_set, paired_bootstrap)
from src.dataset import IMAGE_SIZE, MEAN, STD, LetterboxSquare, load_split, make_loader
from src.evaluate import load_model
from src.train import predict
from src.utils import (DEFAULT_MANIFEST, DEFAULT_OUTPUT_ROOT, get_device, load_json, resolve_data_root,
                       save_json, set_seed)

SEEDS = (42, 43, 44)
FAMILIES = {"orig": "densenet121_pad_seed{seed}", "aug": "densenet121_padaug_seed{seed}"}
PRIOR_AUDIT = {"orig": "perturbation_audit_densenet", "aug": "perturbation_audit_padaug"}
FILLS = {"black": 0, "grey128": 128}
CONDITIONS = (["control", "pad_imgmean"]
              + [f"margin{f}_{k}" for f in (5, 10) for k in ("black", "grey128", "imgmean")]
              + ["pad128_margin128_10", "padmean_margin10mean", "trim5", "trim10", "ring10_interior"])
REUSED = {"pad_imgmean": "pad_imgmean", "margin5_imgmean": "margin5_imgmean", "margin10_imgmean": "margin10_imgmean",
          "padmean_margin10mean": "pad_margin10_imgmean", "ring10_interior": "ring10_interior"}  # ours -> earlier name
AUC_TOL, REG_TOL, TENSOR_TOL = 1e-4, 1e-4, 1e-6


def build_image(gray, cond):
    """Return (perturbed grayscale PIL image, letterbox fill value)."""
    v = int(round(np.asarray(gray).mean()))
    fills = {**FILLS, "imgmean": v}
    if cond == "control":
        return gray, 0
    if cond == "pad_imgmean":
        return gray, v
    if cond.startswith("margin"):
        frac, kind = cond[len("margin"):].split("_", 1)
        return mask_margin(gray, int(frac) / 100, fills[kind]), 0
    if cond == "pad128_margin128_10":
        return mask_margin(gray, 0.10, 128), 128
    if cond == "padmean_margin10mean":
        return mask_margin(gray, 0.10, v), v
    if cond.startswith("trim"):
        w, h = gray.size
        m = round(int(cond[len("trim"):]) / 100 * min(w, h))
        return gray.crop((m, m, w - m, h - m)), 0
    if cond == "ring10_interior":
        return mask_ring(gray, 0.20, 0.10, v), 0
    raise ValueError(cond)


def to_tensor(gray, cond):
    g, fill = build_image(gray, cond)
    return TF.normalize(TF.to_tensor(LetterboxSquare(IMAGE_SIZE, fill=fill)(GRAY3(g))), MEAN, STD)


def fast_metrics(y, p, thr):
    pred = p >= thr
    pos, neg = y == 1, y == 0
    return (float(roc_auc_score(y, p)), float((pred & pos).sum() / pos.sum()), float((~pred & neg).sum() / neg.sum()))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Validation-only margin-sensitivity audit (inference only).")
    ap.add_argument("--data-root")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--audit-dir", default=str(DEFAULT_OUTPUT_ROOT / "shortcut_audit"))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "margin_sensitivity_audit"))
    ap.add_argument("--batch-size", type=int, default=32, help="keep 32 so control matches the saved predictions")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--extra-family-name", default=None,
                    help="OPTIONAL third model family to audit alongside 'orig' and 'aug' (e.g. 'edgeaug'). "
                         "Must be given together with --extra-family-template. Leaving both unset "
                         "reproduces the original two-family ('orig'/'aug') behaviour exactly.")
    ap.add_argument("--extra-family-template", default=None,
                    help="run-folder name template with {seed} for the extra family, e.g. "
                         "'densenet121_edgeaug_seed{seed}'")
    ap.add_argument("--extra-family-pad-aug", choices=("none", "random"), default="random",
                    help="expected pad_aug mode of the extra family's checkpoints (default 'random', "
                         "matching the existing padding-augmented family)")
    ap.add_argument("--extra-family-edge-aug", choices=("none", "random"), default="random",
                    help="expected edge_aug mode of the extra family's checkpoints (default 'random')")
    ap.add_argument("--extra-family-prior-audit", default=None,
                    help="OPTIONAL run folder under runs/ (e.g. 'perturbation_audit_edgeaug') holding an "
                         "earlier perturbation audit for the extra family, used only for the regression "
                         "check against reused conditions. If omitted, that check is skipped for the "
                         "extra family (printed as a NOTE, not an error).")
    args = ap.parse_args(argv)
    if bool(args.extra_family_name) != bool(args.extra_family_template):
        sys.exit("ERROR: --extra-family-name and --extra-family-template must be given together")
    if args.extra_family_name in ("orig", "aug"):
        sys.exit("ERROR: --extra-family-name must not collide with the built-in 'orig'/'aug' families")

    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"ERROR: {out} already contains results; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    root = resolve_data_root(args.data_root)
    device = get_device(args.device)
    amp = device.type == "cuda"
    print("Validation split only; the test split is not read. Inference only; no training.")

    # ---- family configuration: built-in 'orig'/'aug' are unchanged; the extra family is purely additive
    families = dict(FAMILIES)
    prior_audit = dict(PRIOR_AUDIT)
    expected_cfg = {"orig": {"pad_aug": "none", "edge_aug": "none"},
                    "aug": {"pad_aug": "random", "edge_aug": "none"}}
    extra_fam = args.extra_family_name
    if extra_fam:
        families[extra_fam] = args.extra_family_template
        expected_cfg[extra_fam] = {"pad_aug": args.extra_family_pad_aug, "edge_aug": args.extra_family_edge_aug}
        if args.extra_family_prior_audit:
            prior_audit[extra_fam] = args.extra_family_prior_audit
        print(f"Extra family configured: {extra_fam!r} = {args.extra_family_template!r} "
              f"(pad_aug={args.extra_family_pad_aug}, edge_aug={args.extra_family_edge_aug}, "
              f"prior_audit={args.extra_family_prior_audit!r})")

    # ---- alignment of manifest / metadata / images
    meta = pd.read_csv(Path(args.audit_dir) / "metadata_val.csv")
    train_meta = pd.read_csv(Path(args.audit_dir) / "metadata_train.csv")
    frame = load_split(args.manifest, root, "val")
    allm.check_alignment(meta, train_meta, frame)
    order = frame.rel_path.tolist()
    meta = meta.set_index("path").loc[order].reset_index()
    y, patient = meta.target.values, meta.patient_id.values
    grays = [load_gray(p) for p in frame.abs_path]
    sizes = np.array([g.size for g in grays])
    if not ((sizes[:, 0] == meta.width.values) & (sizes[:, 1] == meta.height.values)).all():
        raise RuntimeError("image sizes on disk disagree with metadata_val.csv")
    print(f"Alignment OK: {len(frame)} val images match the manifest by path, label and patient_id; sizes match.")

    # ---- the six checkpoints: verify config, reproduce saved AUC with the saved pipeline
    cks = {}
    ds_ref = None
    align_rows = []
    for fam, tmpl in families.items():
        for seed in SEEDS:
            run = tmpl.format(seed=seed)
            run_dir = DEFAULT_OUTPUT_ROOT / run
            cfg = load_json(run_dir / "config.json")
            model, ckpt = load_model(run_dir / "best.pt", device)
            pre = ckpt["model"].get("preprocess", "resize")
            pa = (ckpt["model"].get("pad_aug") or {}).get("mode", "none")
            ea = (ckpt["model"].get("edge_aug") or {}).get("mode", "none")
            want_pa, want_ea = expected_cfg[fam]["pad_aug"], expected_cfg[fam]["edge_aug"]
            if (ckpt["model"]["name"] != "densenet121" or cfg["model"] != "densenet121" or pre != "pad"
                    or cfg["preprocess"] != pre or ckpt.get("seed") != seed or cfg["seed"] != seed
                    or cfg.get("pretrained") or pa != want_pa or cfg.get("pad_aug", "none") != want_pa
                    or ea != want_ea or cfg.get("edge_aug", "none") != want_ea):
                raise RuntimeError(f"{run}: unexpected configuration")
            model.eval()
            loader, fr = make_loader(args.manifest, root, "val", args.batch_size, num_workers=0, seed=args.seed,
                                     train=False, pin_memory=device.type == "cuda", preprocess=pre)
            if fr.rel_path.tolist() != order:
                raise RuntimeError(f"{run}: loader order differs from metadata order")
            ds_ref = loader.dataset
            ref, tg, _ = predict(model, loader, device, amp=amp, desc=f"{fam}{seed} reference")
            if not (tg == y).all():
                raise RuntimeError(f"{run}: label order mismatch")
            saved = load_json(run_dir / "val_metrics.json")["roc_auc"]
            auc = roc_auc_score(tg, ref)
            if abs(auc - saved) > AUC_TOL:
                raise RuntimeError(f"{run}: reference AUC {auc:.6f} != saved {saved:.6f}")
            thr = load_json(run_dir / "threshold.json")["threshold"]
            cks[(fam, seed)] = {"model": model, "thr": thr, "ref": ref, "run": run}
            align_rows.append({"family": fam, "seed": seed, "run": run, "preprocess": pre, "pad_aug_mode": pa,
                               "edge_aug_mode": ea, "saved_threshold": thr, "reference_pooled_auc": auc,
                               "saved_pooled_auc": saved, "auc_abs_diff": abs(auc - saved)})
            print(f"{fam} seed {seed} ({run}): densenet121/pad/pad_aug={pa}/edge_aug={ea}; reference AUC "
                  f"{auc:.6f} vs saved {saved:.6f} OK; saved threshold {thr:.4f}")

    # ---- conditions: build tensors once, evaluate all six checkpoints
    probs = {}
    ctrl_stack = None
    tgt = torch.tensor(y, dtype=torch.long)
    changed_area = {}
    for cond in CONDITIONS:
        stack = torch.stack([to_tensor(g, cond) for g in tqdm(grays, desc=f"build {cond}", leave=False)])
        if cond == "control":
            ctrl_stack = stack
            ref_t = torch.stack([ds_ref[i][0] for i in range(len(ds_ref))])
            if not torch.equal(stack, ref_t):
                raise RuntimeError("control tensors differ from the saved dataset pipeline's tensors")
            del ref_t
            print("control tensors are bit-identical to the saved pipeline's tensors for all 878 images.")
        changed_area[cond] = float(((stack - ctrl_stack).abs().amax(dim=1) > TENSOR_TOL).float().mean())
        loader = DataLoader(TensorDataset(stack, tgt), batch_size=args.batch_size, shuffle=False)
        for key, c in cks.items():
            pr, tg, _ = predict(c["model"], loader, device, amp=amp, desc=f"{key[0]}{key[1]} {cond}")
            if not (tg == y).all():
                raise RuntimeError("target order changed")
            probs[(key, cond)] = pr
        del stack, loader
    align = pd.DataFrame(align_rows)
    align["control_vs_saved_pipeline_max_abs_dprob"] = [
        float(np.abs(probs[(k, "control")] - cks[k]["ref"]).max()) for k in cks]
    align["changed_canvas_frac_by_condition"] = ""
    align.to_csv(out / "alignment_checks.csv", index=False)
    if (align.control_vs_saved_pipeline_max_abs_dprob > REG_TOL).any():
        raise RuntimeError("control predictions differ from the saved-pipeline predictions")
    pd.DataFrame([{"condition": c, "mean_canvas_fraction_changed_vs_control": v} for c, v in changed_area.items()]
                 ).to_csv(out / "condition_canvas_change.csv", index=False)
    print("control predictions match the saved pipeline for all six checkpoints.")

    # ---- regression against the earlier audits (reused conditions)
    reg = []
    for fam in families:
        if fam not in prior_audit:
            print(f"NOTE: no prior-audit configured for family {fam!r}; skipping its regression check "
                  "(pass --extra-family-prior-audit to enable it).")
            continue
        prior = pd.read_csv(DEFAULT_OUTPUT_ROOT / prior_audit[fam] / "per_image_predictions.csv")
        for seed in SEEDS:
            for ours, theirs in REUSED.items():
                p0 = prior[(prior.seed == seed) & (prior.condition == theirs)].sort_values("val_index")
                if len(p0) != len(y) or p0.path.tolist() != order:
                    reg.append({"family": fam, "seed": seed, "condition": ours, "earlier_condition": theirs,
                                "n": len(p0), "max_abs_dprob": float("nan"), "passed": False})
                    continue
                d = float(np.abs(p0.prob.values - probs[((fam, seed), ours)]).max())
                reg.append({"family": fam, "seed": seed, "condition": ours, "earlier_condition": theirs,
                            "n": len(p0), "max_abs_dprob": d, "passed": d <= REG_TOL})
    reg = pd.DataFrame(reg)
    reg.to_csv(out / "regression_checks.csv", index=False)
    print(f"regression checks vs earlier audits: {int(reg.passed.sum())}/{len(reg)} passed "
          f"(max |dp| {reg.max_abs_dprob.max():.2e}, tolerance {REG_TOL})")

    # ---- metrics per checkpoint and condition
    rows, per_image = [], []
    for key, c in cks.items():
        fam, seed = key
        thr, pc = c["thr"], probs[(key, "control")]
        for cond in CONDITIONS:
            pr = probs[(key, cond)]
            m = metric_set(y, pr, thr)
            ch = (pr >= thr) != (pc >= thr)
            nrm, pn = y == 0, y == 1
            m.update({"family": fam, "seed": seed, "condition": cond, "threshold": thr,
                      "pred_changed": int(ch.sum()), "mean_abs_dprob": float(np.abs(pr - pc).mean()),
                      "normal_mean_dprob": float((pr - pc)[nrm].mean()), "pneumonia_mean_dprob": float((pr - pc)[pn].mean()),
                      "normal_mean_abs_dprob": float(np.abs(pr - pc)[nrm].mean()),
                      "new_false_positives": int((ch & nrm & (pr >= thr)).sum()),
                      "new_false_negatives": int((ch & pn & (pr < thr)).sum())})
            rows.append(m)
            per_image.append(pd.DataFrame({"family": fam, "seed": seed, "condition": cond, "val_index": np.arange(len(y)),
                                           "path": order, "patient_id": patient, "label": meta.label.values,
                                           "prob": pr, "prob_control": pc, "threshold": thr}))
    res = pd.DataFrame(rows)
    res.to_csv(out / "metrics_by_checkpoint_condition.csv", index=False)
    pd.concat(per_image).to_csv(out / "per_image_predictions.csv", index=False)
    show = ["auc", "sensitivity", "specificity", "pred_changed", "new_false_positives", "new_false_negatives",
            "mean_abs_dprob", "normal_mean_dprob"]
    fam_summary = res.groupby(["family", "condition"], sort=False)[show + ["pr_auc", "precision", "f1", "accuracy",
                                                                           "pneumonia_mean_dprob"]].agg(["mean", "std"])
    fam_summary.round(5).to_csv(out / "family_summary_mean_sd.csv")

    # ---- per-checkpoint paired differences (existing patient-clustered paired bootstrap)
    drows = []
    for key, c in tqdm(cks.items(), desc="paired bootstrap", leave=False):
        for cond in CONDITIONS[1:]:
            ci = paired_bootstrap(y, patient, probs[(key, cond)], probs[(key, "control")], c["thr"], args.n_boot, args.seed)
            a, b = metric_set(y, probs[(key, cond)], c["thr"]), metric_set(y, probs[(key, "control")], c["thr"])
            for k, (lo, hi) in ci.items():
                drows.append({"family": key[0], "seed": key[1], "condition": cond, "metric": k, "control": b[k],
                              "perturbed": a[k], "delta": a[k] - b[k], "ci_lo": lo, "ci_hi": hi})
    dd = pd.DataFrame(drows)
    dd.to_csv(out / "paired_differences_vs_control.csv", index=False)

    # ---- shared patient-clustered bootstrap: NORMAL probability shift CI and original-vs-augmented comparison
    ug, inv = np.unique(patient, return_inverse=True)
    by_group = [np.flatnonzero(inv == i) for i in range(len(ug))]
    rng = np.random.RandomState(args.seed)
    metric_names = ("auc", "sensitivity", "specificity", "normal_dprob")
    store = {(fam, cond, mn): [] for fam in families for cond in CONDITIONS[1:] for mn in metric_names}
    nd_store = {(k, cond): [] for k in cks for cond in CONDITIONS[1:]}
    for _ in tqdm(range(args.n_boot), desc="family bootstrap", leave=False):
        idx = np.concatenate([by_group[i] for i in rng.randint(0, len(ug), len(ug))])
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        nrm = yb == 0
        base = {k: fast_metrics(yb, probs[(k, "control")][idx], cks[k]["thr"]) for k in cks}
        for cond in CONDITIONS[1:]:
            per = {fam: {mn: [] for mn in metric_names} for fam in families}
            for k in cks:
                pc = probs[(k, "control")][idx]
                pr = probs[(k, cond)][idx]
                a, s, sp = fast_metrics(yb, pr, cks[k]["thr"])
                nd = float((pr - pc)[nrm].mean())
                nd_store[(k, cond)].append(nd)
                for mn, v in zip(metric_names, (a - base[k][0], s - base[k][1], sp - base[k][2], nd)):
                    per[k[0]][mn].append(v)
            for fam in families:
                for mn in metric_names:
                    store[(fam, cond, mn)].append(float(np.mean(per[fam][mn])))
    nd_rows = [{"family": k[0], "seed": k[1], "condition": cond, "normal_mean_dprob": res[(res.family == k[0]) & (res.seed == k[1]) & (res.condition == cond)].normal_mean_dprob.iloc[0],
                "ci_lo": float(np.percentile(v, 2.5)), "ci_hi": float(np.percentile(v, 97.5))}
               for (k, cond), v in nd_store.items()]
    pd.DataFrame(nd_rows).to_csv(out / "normal_probability_shift_ci.csv", index=False)
    frows = []
    for cond in CONDITIONS[1:]:
        for mn in metric_names:
            o, a = np.array(store[("orig", cond, mn)]), np.array(store[("aug", cond, mn)])
            diff = a - o
            frows.append({"condition": cond, "metric": mn, "orig_mean_delta": float(o.mean()),
                          "orig_ci_lo": float(np.percentile(o, 2.5)), "orig_ci_hi": float(np.percentile(o, 97.5)),
                          "aug_mean_delta": float(a.mean()), "aug_ci_lo": float(np.percentile(a, 2.5)),
                          "aug_ci_hi": float(np.percentile(a, 97.5)), "aug_minus_orig": float(diff.mean()),
                          "diff_ci_lo": float(np.percentile(diff, 2.5)), "diff_ci_hi": float(np.percentile(diff, 97.5))})
    fc = pd.DataFrame(frows)
    fc.to_csv(out / "family_comparison.csv", index=False)

    # ---- OPTIONAL additive comparison: extra family vs 'orig' (does not alter family_comparison.csv)
    fc_extra = None
    if extra_fam:
        erows = []
        for cond in CONDITIONS[1:]:
            for mn in metric_names:
                o, e = np.array(store[("orig", cond, mn)]), np.array(store[(extra_fam, cond, mn)])
                diff = e - o
                erows.append({"condition": cond, "metric": mn, "orig_mean_delta": float(o.mean()),
                              "orig_ci_lo": float(np.percentile(o, 2.5)), "orig_ci_hi": float(np.percentile(o, 97.5)),
                              f"{extra_fam}_mean_delta": float(e.mean()),
                              f"{extra_fam}_ci_lo": float(np.percentile(e, 2.5)),
                              f"{extra_fam}_ci_hi": float(np.percentile(e, 97.5)),
                              f"{extra_fam}_minus_orig": float(diff.mean()),
                              "diff_ci_lo": float(np.percentile(diff, 2.5)), "diff_ci_hi": float(np.percentile(diff, 97.5))})
        fc_extra = pd.DataFrame(erows)
        fc_extra.to_csv(out / f"family_comparison_{extra_fam}_vs_orig.csv", index=False)

    # ---- recurring false-positive images (NORMAL, correct at control, flipped to FP)
    flips = []
    for (key, cond), pr in ((k, probs[k]) for k in probs if k[1] != "control"):
        pc, thr = probs[(key, "control")], cks[key]["thr"]
        f = (y == 0) & (pc < thr) & (pr >= thr)
        flips.append(pd.DataFrame({"family": key[0], "seed": key[1], "condition": cond, "val_index": np.flatnonzero(f)}))
    flips = pd.concat(flips)
    cnt = flips.groupby(["val_index", "family", "condition"]).seed.nunique().unstack(["family", "condition"]).fillna(0).astype(int)
    cnt.columns = [f"{f}_{c}" for f, c in cnt.columns]
    cp = pd.DataFrame({f"ctrl_prob_{fam}_s{s}": probs[((fam, s), "control")] for fam in families for s in SEEDS})
    tab = cnt.join(cp, how="left").join(meta[["path", "patient_id", "width", "height", "aspect", "pixels"]], how="left")
    tab.index.name = "val_index"
    tab["aug_margin_any_flips"] = tab[[c for c in tab.columns if c.startswith("aug_margin") or c.startswith("aug_trim")]].sum(axis=1)
    tab = tab.sort_values("aug_margin10_imgmean" if "aug_margin10_imgmean" in tab else "aug_margin_any_flips", ascending=False)
    tab.to_csv(out / "recurring_false_positive_images.csv")

    # ---- printed summary
    print("\n=== family means (over seeds 42/43/44), fixed saved thresholds ===")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    for fam in families:
        g = res[res.family == fam].groupby("condition", sort=False)[show].mean().round(4)
        print(f"\n-- {fam}"); print(g.to_string())
    print("\n=== family comparison (aug minus orig, seed-averaged delta vs control; patient-bootstrap 95% CI) ===")
    for mn in ("specificity", "normal_dprob", "sensitivity", "auc"):
        s = fc[fc.metric == mn]
        print(f"-- {mn}")
        for r in s.itertuples():
            print(f"   {r.condition:22s} orig {r.orig_mean_delta:+.4f} | aug {r.aug_mean_delta:+.4f} | aug-orig {r.aug_minus_orig:+.4f} [{r.diff_ci_lo:+.4f},{r.diff_ci_hi:+.4f}]")
    if fc_extra is not None:
        print(f"\n=== extra family comparison ({extra_fam} minus orig, seed-averaged delta vs control; "
              "patient-bootstrap 95% CI) ===")
        for mn in ("specificity", "normal_dprob", "sensitivity", "auc"):
            s = fc_extra[fc_extra.metric == mn]
            print(f"-- {mn}")
            for r in s.itertuples():
                print(f"   {r.condition:22s} orig {r.orig_mean_delta:+.4f} | {extra_fam} "
                      f"{getattr(r, f'{extra_fam}_mean_delta'):+.4f} | {extra_fam}-orig "
                      f"{getattr(r, f'{extra_fam}_minus_orig'):+.4f} [{r.diff_ci_lo:+.4f},{r.diff_ci_hi:+.4f}]")
    print("\nrecurring false positives (NORMAL images flipped to FP by augmented models), count by condition:")
    for cond in CONDITIONS[1:]:
        col = f"aug_{cond}"
        if col in tab:
            print(f"   {cond:22s} images flipped in >=1 seed: {(tab[col] >= 1).sum():3d} | >=2 seeds: {(tab[col] >= 2).sum():3d} | all 3: {(tab[col] == 3).sum():3d}")

    # ---- plots
    palette = ["#3b6fb6", "#d1603d", "#2a9d6f", "#8856a7"]
    cols = {fam: palette[i % len(palette)] for i, fam in enumerate(families)}
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    for ax, (col, ttl) in zip(axes, (("specificity", "Specificity at saved threshold"), ("normal_mean_dprob", "NORMAL mean probability shift vs control"), ("sensitivity", "Sensitivity at saved threshold"))):
        for fam in families:
            for seed in SEEDS:
                s = res[(res.family == fam) & (res.seed == seed)].set_index("condition").loc[CONDITIONS]
                ax.plot(range(len(CONDITIONS)), s[col].values, "o-", color=cols[fam], alpha=0.8, lw=1,
                        label=f"{fam}" if seed == 42 else None)
        ax.set_xticks(range(len(CONDITIONS)), CONDITIONS, rotation=60, ha="right", fontsize=7)
        ax.set_title(ttl, fontsize=9); ax.legend(fontsize=7)
    title = ("Original (blue) vs padding-augmented (orange) DenseNet121, seeds 42-44, VALIDATION. Exploratory."
             if not extra_fam else
             f"orig / aug / {extra_fam} DenseNet121, seeds 42-44, VALIDATION. Exploratory.")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(); fig.savefig(out / "metrics_by_condition.png", dpi=140); plt.close(fig)

    top = tab.index[:4].tolist()
    if top:
        show_c = ["control", "pad_imgmean", "margin10_black", "margin10_grey128", "margin10_imgmean", "trim10", "padmean_margin10mean"]
        fig, axes = plt.subplots(len(top), len(show_c), figsize=(2.2 * len(show_c), 2.3 * len(top)), squeeze=False)
        for r_, i in enumerate(top):
            for c_, cond in enumerate(show_c):
                axes[r_][c_].imshow(denorm(to_tensor(grays[i], cond)), cmap="gray", vmin=0, vmax=1)
                axes[r_][c_].set_xticks([]); axes[r_][c_].set_yticks([])
                axes[r_][c_].set_title(cond if r_ == 0 else "", fontsize=6)
            axes[r_][0].set_ylabel(f"val{i:03d}", fontsize=7)
        fig.suptitle("Model inputs for the NORMAL images flipped to false positives most often by the augmented models", fontsize=8)
        fig.tight_layout(); fig.savefig(out / "recurring_fp_examples.png", dpi=130); plt.close(fig)

    save_json({"note": "Exploratory; validation only; inference only; test split never read. Hypotheses H1-H3 are not findings.",
               "families": families, "extra_family": extra_fam,
               "conditions": CONDITIONS, "n_boot": args.n_boot, "thresholds": {f"{k[0]}_s{k[1]}": c["thr"] for k, c in cks.items()},
               "canvas_fraction_changed": changed_area, "regression_passed": int(reg.passed.sum()),
               "regression_total": int(len(reg)), "regression_max_abs_dprob": float(reg.max_abs_dprob.max()),
               "control_max_abs_dprob_vs_saved": float(align.control_vs_saved_pipeline_max_abs_dprob.max())},
              out / "summary.json")
    print(f"\nOutputs written to {out}")
    if not reg.passed.all():
        sys.exit("ERROR: some regression checks FAILED; see regression_checks.csv")


if __name__ == "__main__":
    main()
