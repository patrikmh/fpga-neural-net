"""Independent body-frame sensor and bounded-pose simulation contracts."""

import importlib
import importlib.util
import math
import unittest

import numpy as np


def unit(quaternion):
    value=np.asarray(quaternion,dtype=np.float64)
    return value/np.linalg.norm(value)


def multiply(a,b):
    w,x,y,z=a
    v,i,j,k=b
    return np.array([w*v-x*i-y*j-z*k,w*i+x*v+y*k-z*j,w*j-x*k+y*v+z*i,w*k+x*j-y*i+z*v])


def reference_pose_step(current,target,dt=.0035,max_rate=6):
    current,target=unit(current),unit(target)
    dot=float(np.dot(current,target))
    if dot<0:
        target=-target
        dot=-dot
    half=math.acos(min(1,max(-1,dot)))
    if half<1e-12:
        return target
    fraction=min(1,max_rate*dt/(2*half))
    return unit((math.sin((1-fraction)*half)*current+math.sin(fraction*half)*target)/math.sin(half))


def reference_sensors(previous,current,dt=.0035):
    previous,current=unit(previous),unit(current)
    w,x,y,z=current
    rotation=np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                       [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                       [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
    conjugate=previous*np.array([1,-1,-1,-1])
    delta=unit(multiply(conjugate,current))
    if delta[0]<0:
        delta=-delta
    magnitude=np.linalg.norm(delta[1:])
    gyro=np.zeros(3) if magnitude<1e-14 else delta[1:]/magnitude*(2*math.atan2(magnitude,delta[0])/dt)
    magnetic=45*np.array([-.02069254372432757,.35690462341735035,-.9339116170267592])
    return np.concatenate((rotation.T@np.array([0,0,9.81]),gyro,rotation.T@magnetic))


class OrientationSimulationTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("host.orientation_sim"),"Orientation simulation module must exist")
        simulator=importlib.import_module("host.orientation_sim")
        for name in ("advance_pose","sensors_for_step"):
            self.assertTrue(callable(getattr(simulator,name,None)),f"Orientation simulator must expose {name}")
        return simulator

    def test_static_identity_and_axis_rotations_use_body_frame_gravity_and_magnetic_units(self):
        simulator=self.api()
        poses=([1,0,0,0],[math.sqrt(.5),math.sqrt(.5),0,0],
               [math.sqrt(.5),0,math.sqrt(.5),0],[math.sqrt(.5),0,0,math.sqrt(.5)])
        for index,pose in enumerate(poses):
            with self.subTest(pose=index):
                expected=reference_sensors(pose,pose)
                actual=np.asarray(simulator.sensors_for_step(pose,pose))
                self.assertEqual(actual.shape,(9,))
                np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)
                np.testing.assert_allclose(simulator.sensors_for_step(-np.asarray(pose),pose),expected,atol=1e-10)
        np.testing.assert_allclose(reference_sensors(poses[1],poses[1])[:3],[0,9.81,0],atol=1e-10)
        np.testing.assert_allclose(reference_sensors(poses[2],poses[2])[:3],[-9.81,0,0],atol=1e-10)

    def test_gyro_reports_body_x_rotation_under_nonidentity_world_pose(self):
        simulator=self.api()
        previous=unit([1,0,0,1])
        angle=.014
        current=multiply(previous,[math.cos(angle/2),math.sin(angle/2),0,0])
        actual=np.asarray(simulator.sensors_for_step(previous,current))
        np.testing.assert_allclose(actual[3:6],[angle/.0035,0,0],atol=1e-10)
        np.testing.assert_allclose(simulator.sensors_for_step(previous,-current),actual,atol=1e-10)

    def test_pose_advance_uses_bounded_shortest_arc_and_stays_normalized_near_identity(self):
        simulator=self.api()
        start=[1,0,0,0]
        target=unit([1,0,0,1])
        expected=[math.cos(.01),0,0,math.sin(.01)]
        advanced=simulator.advance_pose(start,target,dt=.1,max_rate=.2)
        np.testing.assert_allclose(advanced,expected,atol=1e-12)
        np.testing.assert_allclose(simulator.advance_pose(start,-target,dt=.1,max_rate=.2),expected,atol=1e-12)
        near=unit([1,1e-14,-1e-14,0])
        result=np.asarray(simulator.advance_pose(np.array(start)*3,near*7))
        self.assertTrue(np.isfinite(result).all())
        self.assertAlmostEqual(float(np.linalg.norm(result)),1,places=12)
        np.testing.assert_allclose(result,near,atol=1e-12)

    def test_zero_or_nonfinite_quaternions_are_rejected(self):
        simulator=self.api()
        for bad in ([0,0,0,0],[float("nan"),0,0,1],[float("inf"),0,0,1]):
            for function in (simulator.advance_pose,simulator.sensors_for_step):
                with self.subTest(input=bad,function=function.__name__),self.assertRaises(ValueError):
                    function([1,0,0,0],bad)


if __name__=="__main__":
    unittest.main()
