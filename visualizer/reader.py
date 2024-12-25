import cv2
import numpy as np

from base.main import Base


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

        if image_dir is not None:
            self.image_paths = Base.get_image_paths(image_dir)
            self.num_frames = len(self.image_paths)
            self.image_size = Base.load_image(self.image_paths[0]).shape
        elif images is not None and isinstance(images[0], str):
            self.num_frames = len(images)
            self.image_size = Base.load_image(images[0]).shape
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
        if self.image_dir is not None:
            for image_path in self.image_paths:
                return Base.load_image(image_path)
        elif self.images is not None:
            for image in self.images:
                if isinstance(image, str):
                    return Base.load_image(image)
                else:
                    return image
        elif self.video_path is not None:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                return frame
            self.cap.release()
        raise StopIteration
