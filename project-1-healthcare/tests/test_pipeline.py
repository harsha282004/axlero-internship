"""Smoke tests on synthetic dummy images only (no real data, no downloads).

Run from project-1-healthcare/:  python -m unittest discover -s tests -v
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

from src import evaluate, train  # noqa: E402
from src.dataset import build_transforms, load_split, make_loader, read_manifest  # noqa: E402
from src.metrics import compute_metrics, select_threshold  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import close_logging  # noqa: E402


def make_dummy_dataset(root, per_class=4, backslash=False):
    """Create chest_xray/<orig>/<LABEL>/ dummy JPEGs and a manifest; returns manifest path."""
    rng = np.random.RandomState(0)
    rows, pid = [], 0
    for split in ("train", "val", "test"):
        for label in ("NORMAL", "PNEUMONIA"):
            d = Path(root) / "chest_xray" / split / label
            d.mkdir(parents=True, exist_ok=True)
            for i in range(per_class):
                pid += 1
                name = f"img{pid}.jpeg"
                mode = "RGB" if (label == "PNEUMONIA" and i == 0) else "L"  # include one RGB file
                shape = (40 + 8 * i, 56, 3) if mode == "RGB" else (40 + 8 * i, 56)
                Image.fromarray(rng.randint(0, 255, shape, dtype=np.uint8)).save(d / name)
                rel = f"chest_xray/{split}/{label}/{name}"
                rows.append({"path": rel.replace("/", "\\") if backslash else rel, "label": label,
                             "subtype": "normal" if label == "NORMAL" else "virus",
                             "patient_id": f"p{pid}", "orig_split": split, "split": split})
    manifest = Path(root) / "splits.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


class DatasetTests(unittest.TestCase):
    def test_load_and_shapes_with_windows_separators(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = make_dummy_dataset(tmp, backslash=True)
            loader, frame = make_loader(manifest, tmp, "train", batch_size=4, num_workers=0, train=True)
            x, y = next(iter(loader))
            self.assertEqual(tuple(x.shape), (4, 3, 224, 224))
            self.assertEqual(set(frame.target), {0, 1})
            self.assertEqual(dict(zip(frame.label, frame.target)), {"NORMAL": 0, "PNEUMONIA": 1})

    def test_eval_transform_is_deterministic(self):
        t = build_transforms(train=False)
        im = Image.fromarray(np.random.RandomState(1).randint(0, 255, (50, 60), dtype=np.uint8))
        self.assertTrue(torch.equal(t(im), t(im)))

    def test_same_seed_same_shuffle_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = make_dummy_dataset(tmp)
            orders = []
            for _ in range(2):
                loader, _ = make_loader(manifest, tmp, "train", 2, 0, seed=7, train=True)
                orders.append([y.tolist() for _, y in loader])
            self.assertEqual(orders[0], orders[1])

    def test_missing_files_fail_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = make_dummy_dataset(tmp)
            next(Path(tmp, "chest_xray", "val", "NORMAL").glob("*.jpeg")).unlink()
            with self.assertRaisesRegex(FileNotFoundError, "not found"):
                load_split(manifest, tmp, "val")

    def test_patient_leakage_and_bad_label_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = make_dummy_dataset(tmp)
            df = pd.read_csv(manifest)
            df.loc[0, "patient_id"] = df.loc[len(df) - 1, "patient_id"]  # train patient also in test
            df.to_csv(manifest, index=False)
            with self.assertRaisesRegex(ValueError, "leakage"):
                read_manifest(manifest)
            df = pd.read_csv(manifest)
            df.loc[1, "label"] = "COVID"
            df.to_csv(manifest, index=False)
            with self.assertRaisesRegex(ValueError, "Invalid labels"):
                read_manifest(manifest)


class MetricsTests(unittest.TestCase):
    def test_perfect_and_confusion_matrix(self):
        m = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5)
        self.assertEqual(m["confusion_matrix"], [[2, 0], [0, 2]])
        self.assertEqual((m["sensitivity"], m["specificity"], m["precision"], m["f1"]), (1.0, 1.0, 1.0, 1.0))
        self.assertEqual((m["roc_auc"], m["pr_auc"]), (1.0, 1.0))

    def test_single_class_gives_none_not_crash(self):
        m = compute_metrics([1, 1, 1], [0.9, 0.8, 0.2], 0.5)
        self.assertIsNone(m["roc_auc"])
        self.assertIsNone(m["specificity"])
        self.assertEqual(m["sensitivity"], 2 / 3)
        m0 = compute_metrics([0, 0], [0.9, 0.8], 0.5)  # no positives, all predicted positive
        self.assertIsNone(m0["sensitivity"])
        self.assertEqual(m0["precision"], 0.0)

    def test_threshold_selection(self):
        y = [0, 0, 0, 0, 1, 1, 1, 1]
        p = [0.05, 0.1, 0.2, 0.6, 0.4, 0.7, 0.8, 0.9]
        for strategy in ("youden", "f1", "sensitivity"):
            thr, _ = select_threshold(y, p, strategy, 1.0)
            m = compute_metrics(y, p, thr)
            self.assertGreaterEqual(m["sensitivity"], 0.75, strategy)
        thr, _ = select_threshold(y, p, "sensitivity", 1.0)
        self.assertEqual(compute_metrics(y, p, thr)["sensitivity"], 1.0)
        thr, info = select_threshold([1, 1], [0.3, 0.4], "youden")  # single class -> fallback
        self.assertEqual(thr, 0.5)


class ModelTests(unittest.TestCase):
    def test_forward_shapes(self):
        x = torch.randn(2, 3, 224, 224)
        for name in ("baseline", "densenet121"):
            model = build_model(name).eval()
            with torch.no_grad():
                self.assertEqual(tuple(model(x).shape), (2, 1), name)

    def test_pretrained_not_allowed_for_baseline(self):
        with self.assertRaises(ValueError):
            build_model("baseline", pretrained=True)


class EndToEndSmokeTest(unittest.TestCase):
    def test_train_then_evaluate_on_dummy_data(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.addCleanup(close_logging)  # runs first: release log files before deleting the temp dir
        tmp = tmp_dir.name
        manifest = make_dummy_dataset(tmp)
        out = Path(tmp) / "out"
        run_dir = train.main(["--data-root", tmp, "--manifest", str(manifest), "--model", "baseline",
                              "--epochs", "2", "--batch-size", "4", "--num-workers", "0",
                              "--device", "cpu", "--output-dir", str(out), "--run-name", "smoke"])
        for f in ("best.pt", "history.csv", "history.json", "threshold.json", "val_metrics.json",
                  "config.json", "train.log"):
            self.assertTrue((run_dir / f).is_file(), f)
        thr = json.loads((run_dir / "threshold.json").read_text())
        self.assertEqual(thr["selected_on"], "val")
        metrics = evaluate.main(["--checkpoint", str(run_dir / "best.pt"), "--data-root", tmp,
                                 "--manifest", str(manifest), "--split", "test", "--batch-size", "4",
                                 "--num-workers", "0", "--device", "cpu"])
        self.assertEqual(metrics["n"], 8)
        self.assertTrue((run_dir / "eval_test" / "confusion_matrix.png").is_file())
        self.assertEqual(metrics["threshold"], thr["threshold"])  # test did not alter the threshold
        with self.assertRaises(FileExistsError):  # refuses to overwrite a finished run
            train.main(["--data-root", tmp, "--manifest", str(manifest), "--device", "cpu",
                        "--output-dir", str(out), "--run-name", "smoke"])


if __name__ == "__main__":
    unittest.main()
