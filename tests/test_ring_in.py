import numpy as np

from Libs.Lib_RingIn import Response_Is_Unstable


def test_response_instability_uses_raw_response():
    assert not Response_Is_Unstable(1 + 2j)
    assert Response_Is_Unstable(1e8 + 0j)
    assert Response_Is_Unstable(np.inf + 0j)
