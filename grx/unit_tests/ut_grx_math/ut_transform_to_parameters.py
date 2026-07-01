from sim import ttl_testtools as ttl

#====================================================================
# Running tests using pytest
#====================================================================
import numpy as np
from grx import grx_math


def test_transform_to_parameters():
    m_expected = grx_math.parameters_to_transform(grx_math.TransformParams(x=-3,y=7,z=4,roll=-12, pitch=-56 , yaw=-60))    
    transform_parameters = grx_math.transform_to_parameters(m = m_expected)
    m_result = grx_math.parameters_to_transform(transform_parameters)

    assert np.allclose(m_result, m_expected,    rtol=1e-05, atol=1e-08)


#====================================================================
# Running tests maunally
#====================================================================
if __name__ == "__main__":
    # Run TestSystem tests
    ttl.run_tests_manually()