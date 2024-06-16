import ffmpeg
import cv2
import numpy as np
from PIL import Image


# =========================================
# This class was created by Yota Yamamoto.
# Thanks for sharing this code!
# =========================================
class VideoWriter:
    def __init__(
        self,
        filename,
        framerate=1,
        size=(1920, 1080),
        pix_fmt_in=None,
        pix_fmt_ou="yuv420p",
        quality=28,
    ):
        self.filename = filename
        self.framerate = framerate
        self.out = None
        self.width = size[0]
        self.height = size[1]
        self.maxsize = max(size)
        self.quality = quality
        self.pix_fmt_in = pix_fmt_in
        self.pix_fmt_ou = pix_fmt_ou

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.out is not None:
            self.out.stdin.close()
            self.out.wait()

    def _resize(self, image, max_size, image_size):
        image_height, image_width = image_size
        aspect_ratio = float(image_width) / float(image_height)

        if image_width > image_height:
            new_width = max_size
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = max_size
            new_width = int(new_height * aspect_ratio)

        return cv2.resize(
            np.array(image), (new_width, new_height), interpolation=cv2.INTER_LANCZOS4
        )

    def openpipe(self, size, pix_fmt_in):
        width, height = size
        fps = self.framerate
        quality = self.quality
        output = self.filename

        return (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt=pix_fmt_in,
                s="{}x{}".format(width, height),
                r=fps,
            )
            .output(
                str(output),
                pix_fmt=self.pix_fmt_ou,
                qmax=quality,
                qmin=quality,
                loglevel="quiet",
            )
            .overwrite_output()
            .run_async(pipe_stdin=True, quiet=True)
        )

    def write(self, image):
        self.out.stdin.write(np.array(image).tobytes())

    def update(self, image):
        if isinstance(image, Image.Image):
            orig_width, orig_height = image.size
            pix_fmt = "rgb24"
        elif isinstance(image, np.ndarray):
            orig_height, orig_width = image.shape[:2]
            pix_fmt = "bgr24"
            pass
        else:
            raise ValueError(f"image must be Image or ndarray, but {type(image)}!")

        if orig_width != self.width or orig_height != self.height:
            out_image = self._resize(
                image,
                max_size=self.maxsize,
                image_size=(orig_height, orig_width),
            )
            out_height, out_width, _ = out_image.shape
        else:
            out_image = image
            out_height, out_width = orig_height, orig_width

        out_height = out_height - (out_height % 2)
        out_width = out_width - (out_width % 2)
        out_image = cv2.resize(out_image, (out_width, out_height))

        if self.out is None:
            self.out = self.openpipe((out_width, out_height), pix_fmt_in=pix_fmt)

        self.write(out_image)

    # add public exit method
    def close(self):
        self.__exit__(None, None, None)
