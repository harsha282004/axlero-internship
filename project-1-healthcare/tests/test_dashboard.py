"""Smoke tests for dashboard/ helpers, on synthetic dummy data only (no real dataset, no
real checkpoints, no network). Run from project-1-healthcare/:
    python -m unittest discover -s tests -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard import discovery  # noqa: E402
from dashboard.inference import (IncompatibleCheckpointError, InvalidImageError,  # noqa: E402
                                 load_checkpoint_for_inference, open_uploaded_image, predict_image)
from src import train  # noqa: E402
from src.utils import close_logging  # noqa: E402


def make_dummy_dataset(root, per_class=4):
    rng = np.random.RandomState(0)
    rows, pid = [], 0
    for split in ("train", "val", "test"):
        for label in ("NORMAL", "PNEUMONIA"):
            d = Path(root) / "chest_xray" / split / label
            d.mkdir(parents=True, exist_ok=True)
            for i in range(per_class):
                pid += 1
                name = f"img{pid}.jpeg"
                Image.fromarray(rng.randint(0, 255, (48, 48), dtype=np.uint8)).save(d / name)
                rows.append({"path": f"chest_xray/{split}/{label}/{name}", "label": label,
                            "patient_id": f"p{pid}", "orig_split": split, "split": split})
    manifest = Path(root) / "splits.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


class FakeUpload:
    """Minimal stand-in for Streamlit's UploadedFile: only .getvalue() is used by the dashboard."""

    def __init__(self, data: bytes):
        self._data = data

    def getvalue(self):
        return self._data


class InferenceHelperTests(unittest.TestCase):
    def test_invalid_upload_rejected_without_crashing(self):
        with self.assertRaises(InvalidImageError):
            open_uploaded_image(FakeUpload(b"this is not an image"))

    def test_valid_upload_decodes_to_pil_image(self):
        buf_path = None
        with tempfile.TemporaryDirectory() as tmp:
            buf_path = Path(tmp) / "x.png"
            Image.fromarray(np.zeros((40, 40), dtype=np.uint8)).save(buf_path)
            data = buf_path.read_bytes()
        img = open_uploaded_image(FakeUpload(data))
        self.assertEqual(img.size, (40, 40))

    def test_incompatible_checkpoint_rejected_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            bogus = Path(tmp) / "not_a_checkpoint.pt"
            bogus.write_text("not a torch checkpoint")
            with self.assertRaises(IncompatibleCheckpointError):
                load_checkpoint_for_inference(bogus)

    def test_predict_runs_end_to_end_on_a_freshly_trained_checkpoint(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.addCleanup(close_logging)
        tmp = tmp_dir.name
        manifest = make_dummy_dataset(tmp)
        run_dir = train.main(["--data-root", tmp, "--manifest", str(manifest), "--model", "baseline",
                              "--preprocess", "pad", "--epochs", "1", "--batch-size", "4",
                              "--num-workers", "0", "--device", "cpu", "--output-dir", str(Path(tmp) / "out"),
                              "--run-name", "dash_smoke"])
        img = Image.fromarray(np.random.RandomState(1).randint(0, 255, (50, 60), dtype=np.uint8))
        model, ckpt, device = load_checkpoint_for_inference(run_dir / "best.pt")
        result = predict_image(model, ckpt, device, img)
        self.assertIn(result["predicted_class"], ("NORMAL", "PNEUMONIA"))
        self.assertTrue(0.0 <= result["prob_pneumonia"] <= 1.0)
        self.assertAlmostEqual(result["prob_pneumonia"] + result["prob_normal"], 1.0, places=6)
        self.assertEqual(result["preprocess"], "pad")


class DiscoveryTests(unittest.TestCase):
    """Monkeypatches discovery.RUNS_DIR to a disposable temp tree and clears st.cache_data
    caches around each test, since the cached functions take no arguments and would
    otherwise return a stale result keyed on the first RUNS_DIR ever seen in the process."""

    def setUp(self):
        self._orig_runs_dir = discovery.RUNS_DIR
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        discovery.RUNS_DIR = Path(self.tmp_dir.name)
        for fn in (discovery.list_training_runs, discovery.list_audit_dirs,
                  discovery.list_checkpoint_runs, discovery.run_summary,
                  discovery.family_validation_table, discovery.audit_summary,
                  discovery.load_json_safe, discovery.load_csv_safe):
            fn.clear()

    def tearDown(self):
        discovery.RUNS_DIR = self._orig_runs_dir
        for fn in (discovery.list_training_runs, discovery.list_audit_dirs,
                  discovery.list_checkpoint_runs, discovery.run_summary,
                  discovery.family_validation_table, discovery.audit_summary,
                  discovery.load_json_safe, discovery.load_csv_safe):
            fn.clear()

    def _make_training_run(self, name, model="densenet121", pad_aug="none", edge_aug="none",
                           seed=42, roc_auc=0.99, with_checkpoint=True):
        d = discovery.RUNS_DIR / name
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps(
            {"model": model, "preprocess": "pad", "pad_aug": pad_aug, "edge_aug": edge_aug, "seed": seed}))
        (d / "val_metrics.json").write_text(json.dumps(
            {"roc_auc": roc_auc, "sensitivity": 0.9, "specificity": 0.9, "accuracy": 0.9,
             "precision": 0.9, "f1": 0.9, "pr_auc": 0.99, "confusion_matrix": [[9, 1], [1, 9]]}))
        (d / "threshold.json").write_text(json.dumps({"threshold": 0.5, "strategy": "youden"}))
        pd.DataFrame([{"epoch": 1, "val_roc_auc": roc_auc}]).to_csv(d / "history.csv", index=False)
        if with_checkpoint:
            (d / "best.pt").write_bytes(b"not a real checkpoint, existence is all that's checked")
        return d

    def _make_audit_dir(self, name, note="exploratory"):
        d = discovery.RUNS_DIR / name
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps({"note": note, "n_boot": 100}))
        return d

    def test_training_runs_and_audit_dirs_are_distinguished(self):
        self._make_training_run("fake_run_a")
        self._make_audit_dir("fake_audit_a")
        self.assertEqual(discovery.list_training_runs(), ["fake_run_a"])
        self.assertEqual(discovery.list_audit_dirs(), ["fake_audit_a"])

    def test_checkpoint_runs_requires_best_pt(self):
        self._make_training_run("has_ckpt", with_checkpoint=True)
        self._make_training_run("no_ckpt", with_checkpoint=False)
        self.assertEqual(discovery.list_checkpoint_runs(), ["has_ckpt"])

    def test_run_summary_reads_config_and_metrics(self):
        self._make_training_run("r1", model="baseline", pad_aug="random", seed=7, roc_auc=0.87)
        s = discovery.run_summary("r1")
        self.assertEqual(s["model"], "baseline")
        self.assertEqual(s["pad_aug"], "random")
        self.assertEqual(s["seed"], 7)
        self.assertAlmostEqual(s["val_metrics"]["roc_auc"], 0.87)
        self.assertTrue(s["has_checkpoint"])

    def test_family_validation_table_groups_by_configuration(self):
        self._make_training_run("a1", pad_aug="none", seed=42, roc_auc=0.90)
        self._make_training_run("a2", pad_aug="none", seed=43, roc_auc=0.92)
        self._make_training_run("b1", pad_aug="random", seed=42, roc_auc=0.80)
        fam = discovery.family_validation_table()
        self.assertEqual(len(fam), 2)  # two distinct (model, preprocess, pad_aug, edge_aug) groups
        none_group = fam[fam.pad_aug == "none"].iloc[0]
        self.assertEqual(int(none_group["roc_auc_count"]), 2)
        self.assertAlmostEqual(none_group["roc_auc_mean"], 0.91)

    def test_missing_val_metrics_handled_as_none_not_a_crash(self):
        self._make_training_run("broken", with_checkpoint=True)
        (discovery.RUNS_DIR / "broken" / "val_metrics.json").unlink()
        s = discovery.run_summary("broken")
        self.assertIsNone(s["val_metrics"])


if __name__ == "__main__":
    unittest.main()
