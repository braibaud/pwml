from __future__ import absolute_import

import numpy as np
import tensorflow_hub as hu
import sklearn.preprocessing as skp

from ..utilities import imagehelpers as ih
from ..utilities import httphelpers as hh


class TextEmbedder(object):

    def __init__(self, input_size=None, output_size=128, model_url='https://tfhub.dev/google/nnlm-en-dim128/2'):
        self.input_type = 'text'
        self.input_size = input_size
        self.output_size = output_size
        self.model = hu.load(model_url)

        self._empty = np.array([0]*output_size, np.float64)

    def embed_data(self, data, feature):
        if data is None:
            return self._empty
        
        if self.input_size is not None:
            if len(data) > self.input_size:
                data = data[:self.input_size]

        return skp.normalize(
            self.model([data]), 
            axis=1)[0]


class ImageUrlEmbedder(object):

    def __init__(self, input_size=None, output_size=2048, model_url='https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4'):
        self.input_type = 'image_url'
        self.input_size = input_size
        self.output_size = output_size
        self.model = hu.load(model_url)

        self._empty = np.array([0]*output_size, np.float64)

    def embed_data(self, data, feature):
        if data is None:
            return self._empty
        
        # (down)load image
        # resize image to 299x299
        # pass it through the network
        image_data = ih.image_to_batch_array(
            pil_image=ih.extract_square_portion(
                pil_image=hh.download_image(url=data),
                output_size=(299, 299)),
            rescaled=True)

        return skp.normalize(
            self.model(image_data), 
            axis=1)[0]


class CategoryEmbedder(object):

    def __init__(self, input_size=None, output_size=None):

        self.input_type = 'category'
        self.input_size = input_size
        self.output_size = output_size

    def get_empty(self, feature):
        if self.output_size is None:
            return [0] * len(feature.classes)
        else:
            return [0] * self.output_size

    def embed_data(self, data, feature):
        vect = self.get_empty(feature)

        if data is not None:
            position = feature.classes.index(data)

            if position < len(vect):
                vect[position] = 1

        return np.array(
            vect,
            dtype=np.float64)


class NumericEmbedder(object):

    def __init__(self, default_value=0.0):

        self.input_type = 'numeric'
        self.default_value = default_value

    def embed_data(self, data, feature):
        vect = self.default_value

        if data is not None:
            vect = data

        return np.array(
            vect,
            dtype=np.float64)