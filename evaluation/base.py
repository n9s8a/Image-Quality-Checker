from abc import ABC, abstractmethod

class Evaluator(ABC):
    """
    Abstract base class for evaluators
    """
    @abstractmethod
    def evaluate(self, ranked_list):
        """
        Evaluate a ranked list of images.

        Args:
            ranked_list (list of dict): output of ranking pipeline, must contain 'final_score' and 'file'

        Returns:
            dict: evaluation results (metric_name: value)
        """
        pass
