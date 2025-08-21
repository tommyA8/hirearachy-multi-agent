from abc import ABC, abstractmethod

class AbstractAgent(ABC):
    @abstractmethod
    def build(self):
        pass