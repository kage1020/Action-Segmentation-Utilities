import cv2
import numpy as np

from base.main import get_image_paths, load_image


class VideoReader:
    def __init__(
        self,
        image_dir: str | None = None,
        images: list[str] | list[np.ndarray] | None = None,
        video_path: str | None = None,
    ):
        assert (
            image_dir is not None or images is not None or video_path is not None
        ), "Either `image_dir`, `images`, or `video_path` must be provided"

        self.image_dir = image_dir
        self.images = images
        self.video_path = video_path
        self.index = 0

        if image_dir is not None:
            self.image_paths = get_image_paths(image_dir)
            self.num_frames = len(self.image_paths)
            self.image_size = load_image(self.image_paths[0]).shape
        elif images is not None and isinstance(images[0], str):
            self.num_frames = len(images)
            self.image_size = load_image(images[0]).shape
        elif images is not None and isinstance(images[0], np.ndarray):
            self.num_frames = len(images)
            self.image_size = images[0].shape
        elif video_path is not None:
            self.cap = cv2.VideoCapture(video_path)
            self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.image_size = (
                int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.num_frames:
            raise StopIteration

        if self.image_dir is not None:
            self.index += 1
            return load_image(self.image_paths[self.index - 1])
        elif self.images is not None:
            self.index += 1
            image = self.images[self.index - 1]
            return load_image(image) if isinstance(image, str) else image
        elif self.video_path is not None:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                return frame
            self.cap.release()
        raise StopIteration
