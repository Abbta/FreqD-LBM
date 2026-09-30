import numpy as np

from Libs.Lib_RingIn import Describe_RingIn_Stop, Response_Is_Unstable


def test_response_instability_uses_raw_response():
    assert not Response_Is_Unstable(1 + 2j)
    assert Response_Is_Unstable(1e8 + 0j)
    assert Response_Is_Unstable(np.inf + 0j)


def test_ring_in_stop_diagnostic_distinguishes_raw_response_and_time_limit():
    def message(response, step):
        return Describe_RingIn_Stop(response, step, 10, 100, True,
                                    complex(np.nan), 3, complex(np.nan), complex(np.nan))

    assert message(1 + 2j, 100) is None
    unstable = message(np.inf + 0j, 100)
    assert 'reason=raw_response_unstable' in unstable
    assert 'reference=True step=100 tbytRI=1' in unstable
    assert 'fit_response=(nan+0j)' in unstable
    timeout = message(1 + 2j, 10001)
    assert 'reason=max_ring_in_time_exceeded' in timeout
    assert 'raw_response=(1+2j)' in timeout
    both = message(np.inf + 0j, 10001)
    assert 'reason=raw_response_unstable+max_ring_in_time_exceeded' in both
