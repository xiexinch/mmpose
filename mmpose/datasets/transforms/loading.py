# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import cv2
import numpy as np
from mmcv.transforms import BaseTransform, LoadImageFromFile

from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadImage(LoadImageFromFile):
    """Load an image from file or from the np.ndarray in ``results['img']``.

    Required Keys:

        - img_path
        - img (optional)

    Modified Keys:

        - img
        - img_shape
        - ori_shape
        - img_path (optional)

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:``mmcv.imfrombytes``.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :func:``mmcv.imfrombytes`` for details.
            Defaults to 'cv2'.
        backend_args (dict, optional): Arguments to instantiate the preifx of
            uri corresponding backend. Defaults to None.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
    """

    def transform(self, results: dict) -> Optional[dict]:
        """The transform function of :class:`LoadImage`.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        try:
            if 'img' not in results:
                # Load image from file by :meth:`LoadImageFromFile.transform`
                results = super().transform(results)
            else:
                img = results['img']
                assert isinstance(img, np.ndarray)
                if self.to_float32:
                    img = img.astype(np.float32)

                if 'img_path' not in results:
                    results['img_path'] = None
                results['img_shape'] = img.shape[:2]
                results['ori_shape'] = img.shape[:2]
        except Exception as e:
            e = type(e)(
                f'`{str(e)}` occurs when loading `{results["img_path"]}`.'
                'Please check whether the file exists.')
            raise e

        return results


@TRANSFORMS.register_module()
class LoadImageFromOSS2(BaseTransform):

    def __init__(self, endpoint: str, ak: str, sk: str, bucket_name: str,
                 **kwargs):
        super().__init__(**kwargs)
        self.endpoint = endpoint
        self.ak = ak
        self.sk = sk
        self.bucket_name = bucket_name

    def transform(self, results: dict) -> Optional[dict]:
        """The transform function of :class:`LoadImage`.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        try:
            import oss2
            auth = oss2.Auth(self.ak, self.sk)
            bucket = oss2.Bucket(auth, self.endpoint, self.bucket_name)
            img_path = results['img_path']
            img = bucket.get_object(img_path).read()
            img = np.frombuffer(img, np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            results['img'] = img
            results['img_shape'] = img.shape[:2]
            results['ori_shape'] = img.shape[:2]
        except ImportError:
            raise ImportError('Please install oss2 to load image from oss')

        return results
