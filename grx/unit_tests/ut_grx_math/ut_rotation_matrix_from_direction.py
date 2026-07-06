import sys
import os
from pathlib import Path
from sim import ttl_testtools as ttl

#====================================================================
# Running tests using pytest
#====================================================================
import numpy as np
from grx import grx_math

# ^ Y (up)            
# |              
# |              
# | . Z (front)         
# |/             
# +-------> X (right)  

def test_rotation_matrix_from_direction():
    front = np.array([1,0,1])       # front-right
    strive_up = np.array([0,1,0])   # up
    r_result = grx_math.rotation_matrix_from_direction(front = front,  strive_up=strive_up  )
    r_expected = np.array([ [ 0.70710678,  0,  0.70710678,          0        ],                            
                            [ 0,           1,           0,          0        ], 
                            [ -0.70710678, 0,  0.70710678,          0        ], 
                            [ 0,           0,           0,          1        ]
                          ])
    assert np.allclose(r_result, r_expected ,   rtol=1e-05, atol=1e-08)


#====================================================================
# Running tests maunally
#====================================================================
if __name__ == "__main__":
    # Run TestSystem tests
    ttl.run_tests_manually()