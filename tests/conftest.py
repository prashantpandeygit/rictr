import pytest
import torch


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)

@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

