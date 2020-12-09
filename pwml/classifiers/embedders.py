from __future__ import absolute_import

import os
import numpy as np
import tensorflow_hub as hu
import sklearn.preprocessing as skp
import shelve as she

from ..utilities import imagehelpers as ih
from ..utilities import httphelpers as hh
from ..utilities import filehelpers as fh

NNLM_EN_DIM128 = {
    'input_size': None,
    'input_type': 'text',
    'output_size': 128,
    'model_url': 'https://tfhub.dev/google/nnlm-en-dim128/2'
}

UNIVERSAL_SENTENCE_ENCODER_LARGE = {
    'input_size': None,
    'input_type': 'longtext',
    'output_size': 512,
    'model_url': 'https://tfhub.dev/google/universal-sentence-encoder-large/5'
}

IMAGENET_INCEPTION_V3_FEATURE_VECTOR = {
    'input_size': (299, 299),
    'input_type': 'image_url',
    'output_size': 2048,
    'model_url': 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4'
}


class BaseEmbedder:

    def __init__(self, input_type=None, input_size=None, output_size=None, cache_path=None):

        self.input_type = input_type
        self.input_size = input_size
        self.output_size = output_size
        self.cache_path = cache_path
        self._shelve = None

        self.load_cache()

    def __exit__(self, *exc_info):
        self.save_cache()

    def load_cache(self):
        
        cache_path = None

        if os.path.exists(self.cache_path):
            cache_path = os.path.join(
                self.cache_path, 
                self.input_type)
        else:
            cache_path = self.input_type

        self._shelve = she.open(
            filename=cache_path)

    def save_cache(self):
        self._shelve.close()

    def flush_cache(self):
        self.save_cache()
        self.load_cache()

    def data_empty(self, data, feature):
        return data

    def data_preparation(self, data, feature):
        return data

    def data_embedding(self, data, feature):
        return data

    def embed_data(self, data, feature):
        if data is None:
            return self.data_empty(
                data=data, 
                feature=feature)
        
        data_prepared = self.data_preparation(
            data=data, 
            feature=feature)

        embedding = None

        if data in self._shelve:
            embedding = self._shelve[data]
        else:
            embedding = self.data_embedding(
                data=data_prepared, 
                feature=feature)
            
            self._shelve[data] = embedding
        
        return embedding


class CategoryEmbedder(BaseEmbedder):

    def __init__(self, input_size=None, output_size=None, cache_path=None):
        super().__init__(
            input_type='category',
            input_size=input_size,
            output_size=output_size,
            cache_path=cache_path)

    def data_empty(self, data, feature):
        empty_data = None

        if self.output_size is None:
            empty_data = [0] * len(feature.classes)
        else:
            empty_data = [0] * self.output_size

        return np.array(
            empty_data,
            dtype=np.float64)

    def data_embedding(self, data, feature):
        vect = self.data_empty(data, feature)

        if data is not None:
            position = feature.classes.index(data)

            if position < len(vect):
                vect[position] = 1

        return vect


class NumericEmbedder(BaseEmbedder):

    def __init__(self, default_value=0.0, cache_path=None):
        super().__init__(
            input_type='numeric',
            input_size=None,
            output_size=None, 
            cache_path=cache_path)

        self.default_value = default_value

    def data_empty(self, data, feature):
        return np.array(
            [self.default_value],
            dtype=np.float64)

    def data_embedding(self, data, feature):
        return np.array(
            [data],
            dtype=np.float64)


class BaseHubEmbedder(BaseEmbedder):

    def __init__(self, input_type=None, input_size=None, output_size=None, model_url=None, cache_path=None):
        super().__init__(
            input_type=input_type,
            input_size=input_size,
            output_size=output_size, 
            cache_path=cache_path)

        self.model = hu.load(model_url)
        self._empty = np.array([0]*output_size, np.float64)

    def data_empty(self, data, feature):
        return self._empty

    def data_embedding(self, data, feature):
        return skp.normalize(
            self.model(data), 
            axis=1)[0]


class TextEmbedder(BaseHubEmbedder):

    def data_preparation(self, data, feature):
        input = data
        if self.input_size is not None and len(data) > self.input_size:
            input = data[:self.input_size]
            
        return np.array([input])


class ImageUrlEmbedder(BaseHubEmbedder):

    def data_preparation(self, data, feature):
        pil_image = hh.download_image(
            url=data)
        
        if pil_image.size != self.input_size:
            pil_image = ih.extract_square_portion(
                pil_image=pil_image,
                output_size=self.input_size)

        return ih.image_to_batch_array(
            pil_image=pil_image,
            rescaled=True)
