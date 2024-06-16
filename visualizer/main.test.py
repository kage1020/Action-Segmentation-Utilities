from .main import Visualizer
import unittest
import numpy as np
import torch


class TestVisualizer(unittest.TestCase):
    def test_load_images(self):
        images = Visualizer.load_images("docs")
        self.assertEqual(len(images), 2)

    def test_load_image(self):
        image = Visualizer.load_image(image_path="docs/feature.png")
        self.assertEqual(isinstance(image, np.ndarray), True)

    def test_to_np(self):
        x = [1, 2, 3]
        self.assertTrue(np.array_equal(Visualizer.to_np(x)[0], np.array(x)))
        x = np.array([1, 2, 3])
        self.assertTrue(np.array_equal(Visualizer.to_np(x)[0], x))
        x = torch.tensor([1, 2, 3])
        self.assertTrue(np.array_equal(Visualizer.to_np(x)[0], x.numpy()))

    def test_to_segments(self):
        x = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        backgrounds = []
        self.assertEqual(
            Visualizer.to_segments(x, backgrounds)[0],
            [(1, (0, 2)), (2, (2, 4)), (3, (4, 6)), (4, (6, 8)), (5, (8, 10))],
        )
        backgrounds = [2]
        self.assertEqual(
            Visualizer.to_segments(x, backgrounds)[0],
            [(1, (0, 2)), (3, (4, 6)), (4, (6, 8)), (5, (8, 10))],
        )
        x = [0, 0, 0, 1, 1, 2, 0, 0, 1, 1, 1, 0, 0]
        backgrounds = [0]
        self.assertEqual(
            Visualizer.to_segments(x, backgrounds)[0],
            [(1, (3, 5)), (2, (5, 6)), (1, (8, 11))],
        )
        backgrounds = [0, 2]
        self.assertEqual(
            Visualizer.to_segments(x, backgrounds)[0], [(1, (3, 5)), (1, (8, 11))]
        )

    def test_str_to_int(self):
        x = ["1", "2", "3"]
        mapping = {"1": 0, "2": 1, "3": 2}
        self.assertEqual(Visualizer.str_to_int(x), ([0, 1, 2], mapping))


if __name__ == "__main__":
    unittest.main()
