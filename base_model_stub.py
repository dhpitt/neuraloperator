import torch

from neuralop.tests.test_utils import DummyModel
from neuralop.models.base_model import BaseModel

class NestedBaseModel(BaseModel, name="NestedBaseModel"):
    def __init__(self, submodel):
        super().__init__()
        self.submodel = submodel
    
nested_model = NestedBaseModel(submodel=DummyModel(10))
torch.save(nested_model.state_dict(), "./nested_state_dict.pt")
loaded_state_dict = torch.load("./nested_state_dict.pt", weights_only=True)
#nested_model.