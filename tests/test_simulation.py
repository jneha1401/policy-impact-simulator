import pytest
import numpy as np
from src.simulation_model import PolicyImpactModel

def test_model_initialization():
    model = PolicyImpactModel()
    assert model is not None
    assert not model.is_trained
