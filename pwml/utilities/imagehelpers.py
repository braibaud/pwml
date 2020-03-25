import math as mth
import numpy as np
import PIL as pil


class Rectangle(object):
    """
    A class to manufacture rectangle objects.
    """

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.width = w
        self.height = h

    def __str__(self):
        return 'Rectangle(x:{0}, y:{1}, width:{2}, height:{3})'.format(
            self.x,
            self.y,
            self.width,
            self.height)

    def to_simple_text(self):
        """
        Returns a simple text version of the rectangle
        """
        return '{0}-{1}-{2}-{3}'.format(
            self.x,
            self.y,
            self.width,
            self.height)

    def lt(self):
        """
        Returns the point at the (left, top) location of the rectangle
        """
        return self.x, self.y

    def rb(self):
        """
        Returns the point at the (right, bottom) location of the rectangle
        """
        return self.x + self.width, self.y + self.height

    def ltrb(self):
        """
        Returns a tuple (left, top, right, bottom) representing the rectangle
        """
        return (*self.lt(), *self.rb())

    def contains(self, other):
        """
        Checks whether the current rectangle contains the other one
        """
        return (other.x >= self.x and
                other.y >= self.y and
                other.x + other.width <= self.width and
                other.y + other.height <= self.height)

    def grow(self, ratio, image_width, image_height):
        """
        Try and inflate a rectangle by a ratio, without exceeding
        the image itself.
        """
        image = Rectangle(0, 0, image_width, image_height)

        w_max = min(
            min(self.x, image.width - (self.x + self.width)),
            mth.floor(self.width * ratio / 2.0))

        h_max = min(
            min(self.y, image.height - (self.y + self.height)),
            mth.floor(self.height * ratio / 2.0))

        r_max = min(
            2 * w_max / self.width,
            2 * h_max / self.height)

        w = mth.floor(self.width * r_max / 2.0)

        h = mth.floor(self.height * r_max / 2.0)

        new_rec = Rectangle(
            self.x - w,
            self.y - h,
            self.width + 2 * w,
            self.height + 2 * h)

        assert image.contains(new_rec)

        return new_rec


def image_to_array(pil_image, rescaled=True):
    """
    Convert a PIL image into a numpy array
    :param pil_image: the PIL image object
    :param rescaled: if True rescales between 0 and 1
    :return: the converted image
    """
    return np.array(pil_image, dtype=np.float32) / (255.0 if rescaled else 1.0)


def image_to_batch_array(pil_image, rescaled=True):
    """
    Convert a PIL image into an image batch
    :param pil_image: the PIL image object
    :param rescaled: if True rescales between 0 and 1
    :return: the converted image batch
    """
    return image_to_array(
        pil_image=pil_image, 
        rescaled=rescaled)[np.newaxis, :, :, :]


def crop_image(pil_image, rectangle):

    return pil_image.transform(
        size=(rectangle.width, rectangle.height),
        method=pil.Image.EXTENT,
        resample=pil.Image.BILINEAR,
        data=rectangle.ltrb())


def extract_square_portion(
    pil_image,  # a PIL Image object
    h_position=None,  # Left, Center or Right (None=Center)
    v_position=None,  # Top, Middle or Bottom (None=Middle)
    output_size=None):  # a tuple (width, height) (None=Max)

    max_dim = min(pil_image.width, pil_image.height)
    x = 0
    y = 0

    if h_position == 'Left':
        x = 0
    elif h_position in ['Center', None]:
        x = mth.floor((pil_image.width - max_dim) / 2)
    elif h_position == 'Right':
        x = pil_image.width - max_dim

    if v_position == 'Top':
        y = 0
    elif v_position in ['Middle', None]:
        y = mth.floor((pil_image.height - max_dim) / 2)
    elif v_position == 'Bottom':
        y = pil_image.height - max_dim

    if output_size == None:
        output_size = (max_dim, max_dim)

    return pil_image.transform(
        size=output_size,
        method=pil.Image.EXTENT,
        resample=pil.Image.BILINEAR,
        data=(x, y, x + max_dim, y + max_dim))
