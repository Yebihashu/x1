from sim import ttl_testtools as ttl

import math
import numpy as np
from grx import grx_math


def test_parameters_to_transform__translation():
    m = grx_math.parameters_to_transform(grx_math.TransformParams(x=1,y=2,z=3,roll=0.0, pitch=0.0, yaw= 0.0))    
    v = np.array([4,-3,0.5,1])
    v_result = m.dot(v)
    v_expected = v + np.array([1,2,3,0])
    assert np.allclose(v_result, v_expected,    rtol=1e-05, atol=1e-08)

def test_parameters_to_transform__rotation_roll():
    m = grx_math.parameters_to_transform(grx_math.TransformParams(x=0,y=0,z=0,roll=45, pitch=0 , yaw=0))    
    v = np.array([1,0,1,1])
    v_result = m.dot(v)
    v_expected = np.array([math.cos(45*math.pi/180),math.sin(45*math.pi/180), 1,1])
    assert np.allclose(v_result, v_expected,    rtol=1e-05, atol=1e-08)

def test_parameters_to_transform__rotation_pitch():
    m = grx_math.parameters_to_transform(grx_math.TransformParams(x=0,y=0,z=0,roll=0, pitch=-45 , yaw=0))    
    v = np.array([1,0,1,1])
    v_result = m.dot(v)
    v_expected = np.array([1, math.sin(45*math.pi/180),math.cos(45*math.pi/180),1])
    assert np.allclose(v_result, v_expected,    rtol=1e-05, atol=1e-08)

def test_parameters_to_transform__rotation_yaw():
    m = grx_math.parameters_to_transform(grx_math.TransformParams(x=0,y=0,z=0,roll=0, pitch=0 , yaw=-60))    
    v = np.array([0,1,1,1])
    v_result = m.dot(v)
    v_expected = np.array([-math.sin(60*math.pi/180), 1, math.cos(60*math.pi/180),1])
    assert np.allclose(v_result, v_expected,    rtol=1e-05, atol=1e-08)

def test_parameters_to_transform__rotation():
    m = grx_math.parameters_to_transform(grx_math.TransformParams(x=0,y=0,z=0,roll=-45, pitch=-90 , yaw=-45))    
    v = np.array([-1,0,0,1])
    v_result = m.dot(v)
    v_expected = np.array([0,0,-1,1])
    assert np.allclose(v_result, v_expected,    rtol=1e-05, atol=1e-08)

def test_parameters_to_transform():
    m = grx_math.parameters_to_transform(grx_math.TransformParams(x=1,y=2,z=3,roll=-45, pitch=-90 , yaw=-45))    
    v = np.array([-1,0,0,1])
    v_result = m.dot(v)
    v_expected = np.array([1,2,2,1])
    assert np.allclose(v_result, v_expected,    rtol=1e-05, atol=1e-08)

#====================================================================
# Running tests maunally
#====================================================================
if __name__ == "__main__":
    # Run TestSystem tests
    ttl.run_tests_manually()