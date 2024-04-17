import numpy as np
import torch
import unittest
from main import Evaluator


class TestEvaluator(unittest.TestCase):
    def test_to_np(self):
        x = [1, 2, 3]
        self.assertTrue(np.array_equal(Evaluator.to_np(x), np.array(x)))
        x = np.array([1, 2, 3])
        self.assertTrue(np.array_equal(Evaluator.to_np(x), x))
        x = torch.tensor([1, 2, 3])
        self.assertTrue(np.array_equal(Evaluator.to_np(x), x.numpy()))

    def test_to_segments(self):
        x = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        backgrounds = []
        self.assertEqual(Evaluator.to_segments(x, backgrounds), [(1, (0, 2)), (2, (2, 4)), (3, (4, 6)), (4, (6, 8)), (5, (8, 10))])
        backgrounds = [2]
        self.assertEqual(Evaluator.to_segments(x, backgrounds), [(1, (0, 2)), (3, (4, 6)), (4, (6, 8)), (5, (8, 10))])
        x = [0, 0, 0, 1, 1, 2, 0, 0, 1, 1, 1, 0, 0]
        backgrounds = [0]
        self.assertEqual(Evaluator.to_segments(x, backgrounds), [(1, (3, 5)), (2, (5, 6)), (1, (8, 11))])
        backgrounds = [0, 2]
        self.assertEqual(Evaluator.to_segments(x, backgrounds), [(1, (3, 5)), (1, (8, 11))])

    def test_accuracy_frame(self):
        gt = [1, 2, 3]
        pred = [1, 2, 3]
        self.assertEqual(Evaluator.accuracy_frame(gt, pred), 1.0)
        pred = [1, 2, 4]
        self.assertAlmostEqual(Evaluator.accuracy_frame(gt, pred), 2 / 3)
        pred = [1, 3, 4]
        self.assertAlmostEqual(Evaluator.accuracy_frame(gt, pred), 1 / 3)
        pred = [1, 3, 2]
        self.assertAlmostEqual(Evaluator.accuracy_frame(gt, pred), 1 / 3)
        pred = [3, 1, 2]
        self.assertEqual(Evaluator.accuracy_frame(gt, pred), 0.0)

    def test_accuracy_class(self):
        gt = [1, 1, 2, 2, 3, 3]
        pred = [1, 1, 2, 2, 3, 3]
        self.assertEqual(Evaluator.accuracy_class(gt, pred), [1.0, 1.0, 1.0])
        pred = [1, 1, 2, 3, 3, 3]
        self.assertEqual(Evaluator.accuracy_class(gt, pred), [1.0, 0.5, 1.0])
        pred = [1, 1, 2, 4, 3, 3]
        self.assertEqual(Evaluator.accuracy_class(gt, pred), [1.0, 0.5, 1.0, 0.0])
        pred = [1, 1, 3, 3, 2, 2]
        self.assertEqual(Evaluator.accuracy_class(gt, pred), [1.0, 0.0, 0.0])
        pred = [3, 3, 1, 1, 2, 2]
        self.assertEqual(Evaluator.accuracy_class(gt, pred), [0.0, 0.0, 0.0])

    def test_levenshtein(self):
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        self.assertEqual(Evaluator.levenshtein(x, y), 0)
        y = [1, 2, 3, 4, 6]
        self.assertEqual(Evaluator.levenshtein(x, y), 1)
        y = [1, 2, 3, 4]
        self.assertEqual(Evaluator.levenshtein(x, y), 1)
        y = [1, 2, 3, 4, 5, 6]
        self.assertEqual(Evaluator.levenshtein(x, y), 1)
        y = [1, 2, 3, 5]
        self.assertEqual(Evaluator.levenshtein(x, y), 1)
        y = [1, 2, 3, 5, 4]
        self.assertEqual(Evaluator.levenshtein(x, y), 2)
        y = [1, 2, 3, 5, 6]
        self.assertEqual(Evaluator.levenshtein(x, y), 2)

    def test_edit_score(self):
        gt = [0, 0, 1, 1, 1, 2, 3, 3, 0, 0]
        pred = [0, 0, 1, 1, 1, 2, 3, 3, 0, 0]
        backgrounds = [0]
        self.assertEqual(Evaluator.edit_score(gt, pred, backgrounds), 100.0)
        pred = [0, 0, 1, 1, 1, 3, 3, 3, 0, 0]
        self.assertAlmostEqual(Evaluator.edit_score(gt, pred, backgrounds), 2 / 3 * 100.0)
        pred = [0, 0, 1, 1, 1, 2, 3, 3, 4, 0]
        self.assertEqual(Evaluator.edit_score(gt, pred, backgrounds), 3 / 4 * 100.0)
        pred = [0, 0, 1, 1, 1, 4, 3, 3, 0, 0]
        self.assertAlmostEqual(Evaluator.edit_score(gt, pred, backgrounds), 2 / 3 * 100.0)
        pred = [1, 1, 1, 0, 0, 2, 0, 0, 3, 3]
        self.assertEqual(Evaluator.edit_score(gt, pred, backgrounds), 100.0)

    def test_iou(self):
        gt = (0, 5)
        pred = (0, 5)
        self.assertEqual(Evaluator.iou(gt, pred), 1.0)
        pred = (0, 4)
        self.assertEqual(Evaluator.iou(gt, pred), 4 / 5)
        pred = (1, 5)
        self.assertEqual(Evaluator.iou(gt, pred), 4 / 5)
        pred = (1, 4)
        self.assertEqual(Evaluator.iou(gt, pred), 3 / 5)
        pred = (2, 3)
        self.assertEqual(Evaluator.iou(gt, pred), 1 / 5)
        pred = (5, 6)
        self.assertEqual(Evaluator.iou(gt, pred), 0.0)
        pred = (3, 10)
        self.assertEqual(Evaluator.iou(gt, pred), 2 / 10)


if __name__ == '__main__':
    unittest.main()
