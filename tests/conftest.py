import numpy as np
import pytest

SMILES = [
    "O=C(NCCCn1cnc2ccccc21)c1cc(-c2ccccc2)[nH]n1",
    "O=C(NCCCc1nnc2ccccn12)c1cc(-c2ccccc2)[nH]n1",
    "O=c1ccnc2nc(NCc3ccccc3)[nH]n12",
    "c1ccc(CNc2nc3ccccc3o2)cc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "c1ccccc1",
    "CCO",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "O=C(Nc1nc2nccc(=O)n2[nH]1)c1ccccc1",
]


@pytest.fixture
def smiles():
    return list(SMILES)


@pytest.fixture
def activity():
    # first four are potent, remainder inactive
    return np.array([12.0, 30.0, 45.0, 58.0, 80.0, 95.0, 90.0, 88.0, 76.0, 40.0])
