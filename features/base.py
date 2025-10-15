from abc import ABC, abstractmethod

class FeatureExtractor(ABC):
    """
    Abstract base class for feature extractors.
    All extractors must implement extract(image_path) -> dict
    """
    @abstractmethod
    def extract(self, image_path):
        pass