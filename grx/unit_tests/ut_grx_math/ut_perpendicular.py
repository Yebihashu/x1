from sim import ttl_testtools as ttl

#====================================================================
# Running tests using pytest
#====================================================================
import numpy as np
from grx import grx_math

# The UE4/Carla coords system is as follows,where X is the front
# ^ Y (up)            
# |              
# |              
# | . Z (front)         
# |/             
# +-------> X (right)  

def test_perpendicular():
    x = np.array([0,3,3])
    wish_up = np.array([0,1,0])
    perpendicular_result = grx_math.perpendicular(x,wish_up)
    perpendicular_expected = np.array([0 , 0.70710678         , -0.70710678])
    
    assert np.allclose(perpendicular_result, perpendicular_expected ,   rtol=1e-05, atol=1e-08)


#====================================================================
# Running tests maunally
#====================================================================
if __name__ == "__main__":
    # Run TestSystem tests
    ttl.run_tests_manually()