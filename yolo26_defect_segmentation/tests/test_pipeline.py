import csv
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as cfg
from inference import predict_mask, save_visualization
from metrics import compute_metrics, update_confusion_matrix
from prepare_data import convert_sample, prepare_dataset


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def sample(self, defect_type="Crack"):
        path = self.root / f"MT_{defect_type}" / "Imgs" / "sample.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((32, 32), 128, dtype=np.uint8)).save(path)
        mask_path = path.with_name("mask.png")
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[4:12, 6:14] = 255
        Image.fromarray(mask).save(mask_path)
        return {"image_path": str(path), "mask_path": str(mask_path),
                "defect_type": defect_type, "is_defect": int(defect_type != "Free")}

    def test_defect_ids_and_grayscale_channels(self):
        image, mask = convert_sample(self.sample(), 32)
        self.assertEqual(set(np.unique(mask)), {0, 3})
        self.assertEqual(int((np.asarray(mask) == 3).sum()), 64)
        rgb = np.asarray(image)
        np.testing.assert_array_equal(rgb[..., 0], rgb[..., 1])
        np.testing.assert_array_equal(rgb[..., 1], rgb[..., 2])

    def test_normal_mask_ignores_source_mask(self):
        _, mask = convert_sample(self.sample("Free"), 32)
        self.assertEqual(int(np.asarray(mask).sum()), 0)

    def test_inconsistent_label_and_missing_mask_rejected(self):
        row = self.sample()
        row["is_defect"] = 0
        with self.assertRaises(ValueError):
            convert_sample(row, 32)
        row["is_defect"] = 1
        row["mask_path"] = ""
        with self.assertRaises(ValueError):
            convert_sample(row, 32)

    def test_duplicate_split_rejected_before_writing(self):
        row = self.sample()
        split_dir = self.root / "splits"
        split_dir.mkdir()
        for split in ("train", "val", "test"):
            with (split_dir / f"{split}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
        output = self.root / "output"
        with patch.object(cfg, "SPLIT_DIR", split_dir), self.assertRaises(ValueError):
            prepare_dataset(output, 32)
        self.assertFalse(output.exists())

    def test_metrics_include_false_positive_class_exclude_absent(self):
        matrix = np.zeros((4, 4), dtype=np.int64)
        update_confusion_matrix(matrix, np.array([[0, 1, 0, 2]]),
                                np.array([[0, 1, 1, 0]]))
        self.assertEqual(matrix[1, 0], 1)
        metrics = compute_metrics(matrix)
        self.assertAlmostEqual(metrics["foreground_miou"], 0.25)
        self.assertAlmostEqual(metrics["foreground_dice"], 1 / 3)
        self.assertTrue(np.isnan(metrics["iou"][3]))

    def test_background_only_and_invalid_ids(self):
        matrix = np.zeros((6, 6), dtype=np.int64)
        target = np.zeros((2, 2), dtype=np.uint8)
        update_confusion_matrix(matrix, target, target)
        self.assertEqual(compute_metrics(matrix)["foreground_miou"], 0)
        with self.assertRaises(ValueError):
            update_confusion_matrix(matrix, target, np.full_like(target, 255))

    def test_semantic_output_and_visualization(self):
        image = Image.new("RGB", (32, 32))
        mask = np.full((32, 32), 3, dtype=np.uint8)
        class FakeModel:
            def predict(self, **kwargs):
                return [SimpleNamespace(semantic_mask=SimpleNamespace(data=mask))]
        prediction = predict_mask(FakeModel(), image, 32)
        np.testing.assert_array_equal(prediction, mask)
        path = save_visualization(self.root, "result.png", image, prediction, mask)
        with Image.open(path) as result:
            self.assertEqual(result.size, (96, 32))
        with Image.open(self.root / "predictions/result.png") as result:
            np.testing.assert_array_equal(np.asarray(result), mask)


if __name__ == "__main__":
    unittest.main()
