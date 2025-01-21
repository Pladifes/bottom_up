from abc import ABC, abstractmethod

class BaseDataset(ABC):
    @abstractmethod
    def get_panel_data(self):
        pass

    @abstractmethod
    def get_raw_data(self):
        pass
