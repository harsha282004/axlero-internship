"""Smoke tests on synthetic dummy images only (no real data, no downloads).

Run from project-1-healthcare/:  python -m unittest discover -s tests -v
"""
import gc
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
from src.dataset import (IMAGE_SIZE, MEAN, STD, LetterboxSquare, RandomEdgeMask, RandomPadLetterbox,  # noqa: E402
                         build_transforms, load_split, make_loader, read_manifest)
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


class PreprocessModeTests(unittest.TestCase):
    @staticmethod
    def img(mode, w, h, value=128):
        arr = np.full((h, w, 3) if mode == "RGB" else (h, w), value, dtype=np.uint8)
        return Image.fromarray(arr)

    def test_shapes_for_grayscale_and_rgb_in_both_modes(self):
        for preprocess in ("resize", "pad"):
            t = build_transforms(train=False, preprocess=preprocess)
            for mode in ("L", "RGB"):
                for w, h in ((300, 100), (100, 300), (224, 224), (1654, 1323)):
                    self.assertEqual(tuple(t(self.img(mode, w, h)).shape), (3, 224, 224), (preprocess, mode, w, h))

    def test_default_is_resize_and_unknown_mode_rejected(self):
        im = self.img("L", 300, 100)
        self.assertTrue(torch.equal(build_transforms(False)(im), build_transforms(False, "resize")(im)))
        with self.assertRaises(ValueError):
            build_transforms(False, "crop")

    def test_letterbox_keeps_aspect_and_pads_centred_black(self):
        out = LetterboxSquare()(self.img("L", 400, 100, value=200))
        self.assertEqual(out.size, (IMAGE_SIZE, IMAGE_SIZE))
        a = np.array(out)
        rows = np.flatnonzero(a.max(axis=1) > 0)
        self.assertEqual((rows.min(), rows.max()), (84, 139))  # 224x56 image centred vertically
        self.assertEqual(int(a[:84].max()), 0)
        self.assertEqual(int(a[140:].max()), 0)

    def test_same_normalisation_in_both_modes(self):
        im = self.img("L", 400, 100, value=128)
        resized = build_transforms(False, "resize")(im)
        padded = build_transforms(False, "pad")(im)
        expected_pixel = resized[:, 0, 0]  # constant image -> constant tensor
        self.assertTrue(torch.allclose(padded[:, 112, 112], expected_pixel, atol=1e-5))  # image region
        black = torch.tensor([(0 - m) / s for m, s in zip(MEAN, STD)])
        self.assertTrue(torch.allclose(padded[:, 0, 0], black, atol=1e-5))  # padding = normalised black

    def test_rgb_converted_like_grayscale(self):
        a = build_transforms(False, "pad")(self.img("RGB", 300, 100, value=90))
        b = build_transforms(False, "pad")(self.img("L", 300, 100, value=90))
        self.assertTrue(torch.allclose(a, b, atol=0.02))
        raw = a * torch.tensor(STD).view(3, 1, 1) + torch.tensor(MEAN).view(3, 1, 1)  # undo normalisation
        self.assertTrue(torch.allclose(raw[0], raw[1], atol=1e-5))  # three identical grayscale channels
        self.assertTrue(torch.allclose(raw[0], raw[2], atol=1e-5))

    def test_validation_deterministic_and_training_stochastic(self):
        rng = np.random.RandomState(3)
        im = Image.fromarray(rng.randint(0, 255, (120, 300), dtype=np.uint8))
        for preprocess in ("resize", "pad"):
            ev = build_transforms(False, preprocess)
            self.assertTrue(torch.equal(ev(im), ev(im)))
            tr = build_transforms(True, preprocess)
            torch.manual_seed(0)
            first = tr(im)
            self.assertFalse(torch.equal(first, tr(im)))  # augmentation draws new randomness
            torch.manual_seed(0)
            self.assertTrue(torch.equal(first, tr(im)))  # ...but is reproducible from the seed

    def test_train_and_evaluate_with_pad_records_preprocess(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.addCleanup(close_logging)
        tmp = tmp_dir.name
        manifest = make_dummy_dataset(tmp)
        run_dir = train.main(["--data-root", tmp, "--manifest", str(manifest), "--preprocess", "pad",
                              "--epochs", "1", "--batch-size", "4", "--num-workers", "0", "--device", "cpu",
                              "--output-dir", str(Path(tmp) / "out"), "--run-name", "pad"])
        ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
        self.assertEqual(ckpt["model"]["preprocess"], "pad")
        self.assertEqual(json.loads((run_dir / "config.json").read_text())["preprocess"], "pad")
        m = evaluate.main(["--checkpoint", str(run_dir / "best.pt"), "--data-root", tmp, "--manifest",
                           str(manifest), "--split", "val", "--batch-size", "4", "--num-workers", "0",
                           "--device", "cpu"])
        self.assertEqual(m["n"], 8)


class PadAugTests(unittest.TestCase):
    @staticmethod
    def gray(w, h, seed=0, mode="L"):
        rng = np.random.RandomState(seed)
        shape = (h, w, 3) if mode == "RGB" else (h, w)
        return Image.fromarray(rng.randint(30, 220, shape, dtype=np.uint8), mode)

    @staticmethod
    def legacy(train):
        """The pre-change 'pad' pipeline written out explicitly."""
        from torchvision import transforms as T
        head = [T.Grayscale(num_output_channels=3), LetterboxSquare()]
        tail = [T.ToTensor(), T.Normalize(MEAN, STD)]
        if not train:
            return T.Compose(head + tail)
        aug = [T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
               T.ColorJitter(brightness=0.2, contrast=0.2)]
        return T.Compose(head + aug + tail)

    def test_default_none_is_bit_identical_to_legacy_pipeline(self):
        for mode in ("L", "RGB"):
            for w, h in ((300, 100), (100, 300), (224, 224), (517, 331)):
                im = self.gray(w, h, mode=mode)
                self.assertTrue(torch.equal(build_transforms(False, "pad")(im), self.legacy(False)(im)))
                self.assertTrue(torch.equal(build_transforms(False, "pad", "none")(im), self.legacy(False)(im)))
                torch.manual_seed(5)
                a = build_transforms(True, "pad")(im)
                torch.manual_seed(5)
                b = self.legacy(True)(im)
                self.assertTrue(torch.equal(a, b))

    def test_eval_ignores_pad_aug_and_stays_deterministic(self):
        im = self.gray(300, 120)
        ev = build_transforms(False, "pad", "random")
        self.assertTrue(torch.equal(ev(im), ev(im)))
        self.assertTrue(torch.equal(ev(im), self.legacy(False)(im)))

    def test_invalid_arguments_rejected(self):
        with self.assertRaises(ValueError):
            build_transforms(True, "resize", "random")
        with self.assertRaises(ValueError):
            build_transforms(True, "pad", "bogus")
        with self.assertRaises(ValueError):
            RandomPadLetterbox(grey_range=(200, 100))
        with self.assertRaises(ValueError):
            RandomPadLetterbox(grey_range=(0, 300))

    def test_content_preserved_exactly_and_never_cropped(self):
        im = self.gray(300, 100).convert("RGB")
        nh = round(100 * 224 / 300)
        top = (224 - nh) // 2
        ref_content = np.array(LetterboxSquare()(im))[top:top + nh]
        mean_v = int(round(np.asarray(im)[..., 0].mean()))
        tf = RandomPadLetterbox()
        offsets, fills = set(), set()
        for seed in range(300):
            torch.manual_seed(seed)
            out = np.array(tf(im))
            self.assertEqual(out.shape, (224, 224, 3))
            found = [y for y in range(224 - nh + 1) if np.array_equal(out[y:y + nh], ref_content)]
            self.assertEqual(len(found), 1, f"seed {seed}: resized content missing or altered")
            offsets.add(found[0])
            pad = np.concatenate([out[:found[0]], out[found[0] + nh:]]).reshape(-1, 3)
            if len(pad):
                self.assertTrue((pad == pad[0, 0]).all())  # constant, identical across channels
                fills.add(int(pad[0, 0]))
        self.assertGreater(len(offsets), 20)
        self.assertIn(0, fills)
        self.assertIn(mean_v, fills)
        self.assertTrue(all(f in (0, mean_v) or 64 <= f <= 192 for f in fills))
        self.assertTrue(any(f not in (0, mean_v) for f in fills))

    def test_standard_probability_and_equal_fill_mix(self):
        im = self.gray(320, 160)
        std_ref = np.array(LetterboxSquare()(im.convert("RGB")))
        mean_v = int(round(np.asarray(im).mean()))
        tf = RandomPadLetterbox()
        n, standard, kinds = 3000, 0, {"black": 0, "mean": 0, "grey": 0}
        for seed in range(n):
            torch.manual_seed(seed)
            out = np.array(tf(im.convert("RGB")))
            if np.array_equal(out, std_ref):
                standard += 1
                continue
            flat = [r for r in range(224) if out[r].min() == out[r].max()]  # padding rows (content rows vary)
            if not flat:
                continue
            v = int(out[flat[0], 0, 0])
            kinds["black" if v == 0 else "mean" if v == mean_v else "grey"] += 1
        self.assertTrue(0.47 < standard / n < 0.56, standard / n)
        augmented = sum(kinds.values())
        for k in kinds:  # equal thirds of the augmented half (loose tolerance)
            self.assertTrue(0.25 < kinds[k] / augmented < 0.42, (k, kinds))

    def test_seeded_reproducibility_across_runs(self):
        im = self.gray(300, 120)
        tf = build_transforms(True, "pad", "random")
        outs = []
        for _ in range(2):
            torch.manual_seed(11)
            outs.append([tf(im) for _ in range(5)])
        for a, b in zip(*outs):
            self.assertTrue(torch.equal(a, b))
        torch.manual_seed(12)
        self.assertFalse(all(torch.equal(a, tf(im)) for a in outs[0]))

    def test_dataloader_workers_reproducible_and_val_unaffected(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = make_dummy_dataset(tmp)
            runs = []
            for _ in range(2):
                loader, _ = make_loader(manifest, tmp, "train", 2, num_workers=2, seed=7, train=True,
                                        preprocess="pad", pad_aug="random")
                runs.append([x.clone() for x, _ in loader])
            for a, b in zip(*runs):
                self.assertTrue(torch.equal(a, b))
            plain, _ = make_loader(manifest, tmp, "train", 2, num_workers=2, seed=7, train=True, preprocess="pad")
            self.assertFalse(all(torch.equal(a, b) for a, (b, _) in zip(runs[0], plain)))
            v1, _ = make_loader(manifest, tmp, "val", 4, num_workers=0, preprocess="pad", pad_aug="random")
            v2, _ = make_loader(manifest, tmp, "val", 4, num_workers=0, preprocess="pad")
            for (a, _), (b, _) in zip(v1, v2):
                self.assertTrue(torch.equal(a, b))
            del loader, plain, v1, v2  # stop persistent workers so Windows can delete the temp images
            gc.collect()

    def test_checkpoint_metadata_keeps_preprocess_pad(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.addCleanup(close_logging)
        tmp = tmp_dir.name
        manifest = make_dummy_dataset(tmp)
        common = ["--data-root", tmp, "--manifest", str(manifest), "--preprocess", "pad", "--epochs", "1",
                  "--batch-size", "4", "--num-workers", "0", "--device", "cpu", "--output-dir", str(Path(tmp) / "out")]
        run_dir = train.main(common + ["--pad-aug", "random", "--pad-grey-range", "80", "170",
                                       "--run-name", "padaug"])
        ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
        self.assertEqual(ckpt["model"]["preprocess"], "pad")
        meta = ckpt["model"]["pad_aug"]
        self.assertEqual((meta["mode"], meta["grey_range"], meta["keep_standard_prob"], meta["applied_to"]),
                         ("random", [80, 170], 0.5, "train_only"))
        cfg = json.loads((run_dir / "config.json").read_text())
        self.assertEqual((cfg["preprocess"], cfg["pad_aug"], cfg["pad_grey_range"]), ("pad", "random", [80, 170]))
        m = evaluate.main(["--checkpoint", str(run_dir / "best.pt"), "--data-root", tmp, "--manifest", str(manifest),
                           "--split", "val", "--batch-size", "4", "--num-workers", "0", "--device", "cpu"])
        self.assertEqual(m["n"], 8)
        default_dir = train.main(common + ["--run-name", "default"])
        c2 = torch.load(default_dir / "best.pt", map_location="cpu", weights_only=True)
        self.assertEqual(c2["model"]["pad_aug"]["mode"], "none")
        self.assertEqual(c2["model"]["preprocess"], "pad")


class EdgeAugTests(unittest.TestCase):
    @staticmethod
    def gray(w, h, seed=0, mode="L"):
        rng = np.random.RandomState(seed)
        shape = (h, w, 3) if mode == "RGB" else (h, w)
        return Image.fromarray(rng.randint(30, 220, shape, dtype=np.uint8), mode)

    def test_invalid_arguments_rejected(self):
        with self.assertRaises(ValueError):
            build_transforms(True, "pad", edge_aug="bogus")
        with self.assertRaises(ValueError):
            RandomEdgeMask(prob=-0.1)
        with self.assertRaises(ValueError):
            RandomEdgeMask(prob=1.1)
        with self.assertRaises(ValueError):
            RandomEdgeMask(max_frac=0.11)
        with self.assertRaises(ValueError):
            RandomEdgeMask(max_frac=-0.01)
        with self.assertRaises(ValueError):
            RandomEdgeMask(grey_fill=-1)
        with self.assertRaises(ValueError):
            RandomEdgeMask(grey_fill=256)

    def test_shape_always_224_across_ops_modes_and_aspects(self):
        for mode in ("RGB",):
            for w, h in ((300, 100), (100, 300), (224, 224), (517, 331), (41, 37)):
                im = self.gray(w, h, mode=mode)
                tf = build_transforms(True, "pad", edge_aug="random", edge_aug_prob=1.0)
                for seed in range(20):
                    torch.manual_seed(seed)
                    self.assertEqual(tuple(tf(im).shape), (3, IMAGE_SIZE, IMAGE_SIZE), (w, h, seed))

    def test_crop_never_creates_invalid_dimensions_and_uses_original_coordinates(self):
        im = self.gray(300, 100).convert("RGB")
        w, h = im.size
        ref = np.array(im)
        for m in (0, 1, 5, 10):  # 10 = round(0.10 * min(300, 100)), the max_frac=0.10 boundary
            out = RandomEdgeMask.apply(im, m, "crop", fill=0)
            self.assertEqual(out.size, (w - 2 * m, h - 2 * m))
            self.assertGreater(out.size[0], 0)
            self.assertGreater(out.size[1], 0)
            # exact original-image-coordinate crop: output equals ref[m:h-m, m:w-m] bit-for-bit
            self.assertTrue(np.array_equal(np.array(out), ref[m:h - m, m:w - m]))

    def test_mask_band_is_exact_constant_and_interior_untouched(self):
        im = self.gray(320, 160).convert("RGB")
        w, h = im.size
        ref = np.array(im)
        for m in (0, 1, 7, 16):  # 16 = round(0.10 * min(320, 160))
            for fill in (0, 140, 200):
                out = np.array(RandomEdgeMask.apply(im, m, "mask", fill=fill))
                self.assertEqual(out.shape, ref.shape)
                border = np.zeros((h, w), bool)
                border[:m], border[h - m:], border[:, :m], border[:, w - m:] = True, True, True, True
                if m == 0:
                    self.assertTrue(np.array_equal(out, ref))  # m=0 is an exact no-op
                    continue
                # border band is EXACTLY the fill value, identical across channels, on every border pixel
                self.assertTrue((out[border] == fill).all())
                # interior (outside the border) is bit-identical to the original, never touched
                self.assertTrue(np.array_equal(out[~border], ref[~border]))

    def test_fill_choices_are_black_fixed_grey_or_image_mean(self):
        im = self.gray(300, 120).convert("RGB")
        mean_v = int(round(np.asarray(im)[..., 0].mean()))
        mask = RandomEdgeMask(prob=1.0, max_frac=0.10, grey_fill=140)
        fills = set()
        for seed in range(300):
            torch.manual_seed(seed)
            out = np.array(mask(im))
            if out.shape != np.array(im).shape:
                continue
            diff = np.any(out != np.array(im), axis=-1)
            if not diff.any():
                continue
            fills.add(int(out[diff][0, 0]))
        self.assertEqual(fills, {0, 140, mean_v})

    def test_deterministic_evaluation_unaffected_by_edge_aug(self):
        im = self.gray(300, 120)
        legacy_eval = build_transforms(False, "pad")
        ev = build_transforms(False, "pad", edge_aug="random", edge_aug_prob=1.0)
        self.assertTrue(torch.equal(ev(im), ev(im)))
        self.assertTrue(torch.equal(ev(im), legacy_eval(im)))  # eval ignores edge_aug entirely

    def test_default_off_is_bit_identical_to_pipeline_without_the_feature(self):
        for mode in ("L", "RGB"):
            for w, h in ((300, 100), (224, 224), (517, 331)):
                im = self.gray(w, h, mode=mode)
                # eval: identical regardless of edge_aug kwargs being passed at all
                self.assertTrue(torch.equal(build_transforms(False, "pad")(im),
                                            build_transforms(False, "pad", edge_aug="none")(im)))
                # train, default off: identical RNG consumption to the pipeline with no edge_aug kwarg,
                # with and without pad_aug also enabled
                for pad_aug in ("none", "random"):
                    torch.manual_seed(5)
                    a = build_transforms(True, "pad", pad_aug=pad_aug)(im)
                    torch.manual_seed(5)
                    b = build_transforms(True, "pad", pad_aug=pad_aug, edge_aug="none")(im)
                    self.assertTrue(torch.equal(a, b))

    def test_seeded_reproducibility_across_runs(self):
        im = self.gray(300, 120)
        tf = build_transforms(True, "pad", edge_aug="random", edge_aug_prob=0.3)
        outs = []
        for _ in range(2):
            torch.manual_seed(11)
            outs.append([tf(im) for _ in range(8)])
        for a, b in zip(*outs):
            self.assertTrue(torch.equal(a, b))
        torch.manual_seed(12)
        self.assertFalse(all(torch.equal(a, tf(im)) for a in outs[0]))

    def test_composable_with_pad_aug_independently(self):
        im = self.gray(300, 120).convert("RGB")
        tf_both = build_transforms(True, "pad", pad_aug="random", edge_aug="random", edge_aug_prob=1.0)
        torch.manual_seed(1)
        out = tf_both(im)
        self.assertEqual(tuple(out.shape), (3, IMAGE_SIZE, IMAGE_SIZE))

    def test_checkpoint_metadata_records_edge_aug_and_defaults_to_none(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.addCleanup(close_logging)
        tmp = tmp_dir.name
        manifest = make_dummy_dataset(tmp)
        common = ["--data-root", tmp, "--manifest", str(manifest), "--preprocess", "pad", "--pad-aug", "random",
                  "--epochs", "1", "--batch-size", "4", "--num-workers", "0", "--device", "cpu",
                  "--output-dir", str(Path(tmp) / "out")]
        run_dir = train.main(common + ["--edge-aug", "random", "--edge-aug-prob", "0.5", "--edge-max-frac", "0.08",
                                       "--edge-grey-fill", "100", "--run-name", "edgeaug"])
        ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
        meta = ckpt["model"]["edge_aug"]
        self.assertEqual((meta["mode"], meta["prob"], meta["max_frac"], meta["grey_fill"], meta["applied_to"]),
                         ("random", 0.5, 0.08, 100, "train_only"))
        self.assertEqual(ckpt["model"]["pad_aug"]["mode"], "random")  # independent of pad_aug, both recorded
        cfg = json.loads((run_dir / "config.json").read_text())
        self.assertEqual((cfg["edge_aug"], cfg["edge_aug_prob"], cfg["edge_max_frac"], cfg["edge_grey_fill"]),
                         ("random", 0.5, 0.08, 100))
        m = evaluate.main(["--checkpoint", str(run_dir / "best.pt"), "--data-root", tmp, "--manifest", str(manifest),
                           "--split", "val", "--batch-size", "4", "--num-workers", "0", "--device", "cpu"])
        self.assertEqual(m["n"], 8)

        default_dir = train.main(common + ["--run-name", "default"])
        c2 = torch.load(default_dir / "best.pt", map_location="cpu", weights_only=True)
        self.assertEqual(c2["model"]["edge_aug"]["mode"], "none")

    def test_cli_rejects_out_of_range_edge_arguments(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.addCleanup(close_logging)
        tmp = tmp_dir.name
        manifest = make_dummy_dataset(tmp)
        common = ["--data-root", tmp, "--manifest", str(manifest), "--preprocess", "pad", "--epochs", "1",
                  "--batch-size", "4", "--num-workers", "0", "--device", "cpu", "--output-dir", str(Path(tmp) / "out")]
        for bad in (["--edge-aug-prob", "1.5"], ["--edge-max-frac", "0.5"], ["--edge-grey-fill", "300"]):
            with self.assertRaises(ValueError):
                train.main(common + bad + ["--run-name", f"bad{bad[0]}"])


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

        # evaluate.main must not silently overwrite an existing non-empty output directory either
        saved_metrics_before = (run_dir / "eval_test" / "metrics.json").read_text()
        with self.assertRaises(FileExistsError):
            evaluate.main(["--checkpoint", str(run_dir / "best.pt"), "--data-root", tmp,
                           "--manifest", str(manifest), "--split", "test", "--batch-size", "4",
                           "--num-workers", "0", "--device", "cpu"])  # same default eval_test dir, already populated
        self.assertEqual((run_dir / "eval_test" / "metrics.json").read_text(), saved_metrics_before)
        # a fresh --output-dir is unaffected by the guard
        fresh_out = run_dir / "eval_test_rerun"
        metrics2 = evaluate.main(["--checkpoint", str(run_dir / "best.pt"), "--data-root", tmp,
                                  "--manifest", str(manifest), "--split", "test", "--batch-size", "4",
                                  "--num-workers", "0", "--device", "cpu", "--output-dir", str(fresh_out)])
        self.assertEqual(metrics2["n"], 8)
        self.assertTrue((fresh_out / "metrics.json").is_file())


if __name__ == "__main__":
    unittest.main()
