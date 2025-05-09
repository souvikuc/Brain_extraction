from abc import ABC, abstractmethod



class BaseDataLoader(ABC):
    
    @abstractmethod
    def preprocess_data_(self):
        pass
    
    @abstractmethod
    def get_generator_(self):
        pass
    
    @abstractmethod
    def load_data_(self):
        pass
    
    


class BaseModel(ABC):
    
    @abstractmethod
    def load_generator_(self):
        pass
    
    # @abstractmethod
    # def build_model_(self):
    #     pass
    


    
    