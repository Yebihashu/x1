import sys
import os
from pathlib import Path
from sim import ttl_testtools as ttl

#====================================================================
# Running tests using pytest
#====================================================================
import numpy as np
from grx import grx_math



def test_look_at_transform__no_transform():
    transform_results = grx_math.look_at_transform(position_point=np.array([0.0,0.0,0.0]),
                               look_at_point=np.array([0.0,0.0,100.0]), 
                               strive_up=np.array([0.0,1.0,0.0]))
    transform_expected = grx_math.parameters_to_transform(grx_math.TransformParams(x = 0.0, y=0.0, z=0.0, 
                                                                                   roll = 0.0, pitch = 0.0, yaw =0.0 ))
    assert np.allclose(transform_results, transform_expected,    rtol=1e-05, atol=1e-08)

def test_look_at_transform__roll_m90():
    transform_results = grx_math.look_at_transform(position_point=np.array([0.0,0.0,0.0]),
                               look_at_point=np.array([0.0,0.0,100.0]), 
                               strive_up=np.array([1.0,0.0,0.0]))
    transform_expected = grx_math.parameters_to_transform(grx_math.TransformParams(x = 0.0, y=0.0, z=0.0, 
                                                            roll = -90.0, pitch = 0.0, yaw =0.0 ))
    assert np.allclose(transform_results, transform_expected,    rtol=1e-05, atol=1e-08)

def test_look_at_transform__H100_pitch_m45():
    transform_results = grx_math.look_at_transform(position_point=np.array([0.0,100.0,0.0]),
                               look_at_point=np.array([0.0,0.0,100.0]), 
                               strive_up=np.array([0.0,1.0,0.0]))
    transform_expected = grx_math.parameters_to_transform(grx_math.TransformParams(x = 0.0, y=100.0, z=0.0, 
                                                            roll = 0.0, pitch = 45.0, yaw =0.0 ))
    assert np.allclose(transform_results, transform_expected,    rtol=1e-05, atol=1e-08)


def test_look_at_transform__H100_roll_m90_pitch_m45():
    transform_results = grx_math.look_at_transform(position_point=np.array([0.0,100.0,0.0]),
                               look_at_point=np.array([0.0,0.0,100.0]), 
                               strive_up=np.array([-1.0,0.0,0.0]))
    transform_expected = grx_math.parameters_to_transform(grx_math.TransformParams(x = 0.0, y=100.0, z=0.0, 
                                                            roll = 90.0, pitch = 45.0, yaw =0.0 ))
    assert np.allclose(transform_results, transform_expected,    rtol=1e-05, atol=1e-08)


#====================================================================
# Running tests maunally
#====================================================================
if __name__ == "__main__":
    # Run TestSystem tests
    ttl.run_tests_manually()