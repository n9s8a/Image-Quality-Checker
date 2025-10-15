import cv2
import numpy as np
from .base import FeatureExtractor
from utils.io import load_image_pil
from utils.logging import get_logger

logger = get_logger(__name__)

class TechnicalFeatureExtractor(FeatureExtractor):
    """
    Computes technical features for images:
    sharpness, exposure, contrast, face count
    """

    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def _variance_of_laplacian(self, cv_img):
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _exposure_score(self, cv_img):
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        mean = gray.mean() / 255.0
        return np.exp(-((mean - 0.5)**2)/(2*0.18**2))

    def _contrast_score(self, cv_img):
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        std = gray.std() / 255.0
        return 1 - np.exp(-(std**2)/(2*0.12**2))

    def _face_count(self, cv_img):
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30,30))
        return len(faces)

    def extract(self, image_path):
        """
        Compute technical features for an image.

        Args:
            image_path (str): path to image

        Returns:
            dict: {'sharpness', 'exposure', 'contrast', 'faces'}

        Raises:
            RuntimeError on failure.
        """
        try:
            pil_img = load_image_pil(image_path)
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            features = {
                'sharpness': self._variance_of_laplacian(cv_img),
                'exposure': self._exposure_score(cv_img),
                'contrast': self._contrast_score(cv_img),
                'faces': self._face_count(cv_img)
            }
            return features
        except Exception as e:
            logger.exception(f"Failed to extract technical features for {image_path}")
            raise RuntimeError(f"Failed to extract technical features: {e}")
