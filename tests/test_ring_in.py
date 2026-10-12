import numpy as np

from Libs.Lib_RingIn import (
    Describe_RingIn_Stop,
    Latest_Extrapolated_Response,
    Reached_Max_RingIn_Steps,
    Response_Is_Unstable,
)


def test_response_instability_uses_raw_response():
    assert not Response_Is_Unstable(1 + 2j)
    assert not Response_Is_Unstable(1e8 + 0j, response_scale=1e8)
    assert not Response_Is_Unstable(6e8 + 0j, response_scale=1e8)
    assert not Response_Is_Unstable(
        475718629.09448826j,
        response_scale=abs(-53886359.73512857 + 54013315.57565427j),
    )
    assert Response_Is_Unstable(1.1e12 + 0j, response_scale=1e8)
    assert Response_Is_Unstable(np.inf + 0j)


def test_latest_extrapolated_response_uses_fit_history():
    history = np.asarray([1 + 2j, 3 + 4j])
    assert np.isnan(Latest_Extrapolated_Response(history, 0))
    assert Latest_Extrapolated_Response(history, 2) == 3 + 4j


def test_max_ring_in_steps_counts_recorded_steps():
    assert not Reached_Max_RingIn_Steps(0, 0)
    assert not Reached_Max_RingIn_Steps(1998, 2000)
    assert Reached_Max_RingIn_Steps(1999, 2000)


def test_ring_in_stop_diagnostic_distinguishes_raw_response_and_time_limit():
    def message(response, step):
        return Describe_RingIn_Stop(
            response, step, 10, 100, True, complex(np.nan), 3,
            complex(np.nan), complex(np.nan), response_scale=1e8,
        )

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
