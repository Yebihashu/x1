"""
this file test safe_object_bounding  which is a method to get a safe bound around an object in 3D, relative to main camera (cam1) and secondary camera (cam2)

1. the main function defined two cameras cam1 and cam2
*The cameras are defined by sim_cam_m3 class from SIM package. 
*The cameras are pinhole with no distortion. 
*The cameras resolutions are 1080x960 pixels.
*cam1 is located at (0,0,z1) where z is a height of around 1.4 meter 
*cam2 is located at (x2,y2,z2) where:
 x2 is in the range of x1+-0.3 meter
 y2 is in the range of (y1-0.6, y1-1.0) meter
 z2 is in the range of z1+-0.3 meter

 cam1 is looking forward to direciton of [1,0,0] 
 cam2 is lookind fowrad with angle of around 30 degrees to the left (around [1,-1,0])
 
2. the main function defined a cube object with size of LxWxH where:
   L (along x axis) is in the range of (1.0 - 3) meter
   W (along y axis) is in the range of (0.2 0 2.0) meter
   H (along z axis) is in the range of (0.01 - 2.0) meter

   The locaiton of the cube object is set to be in front of cam1 at distance of range of (3.0 - 5.0) meter  


3. the main function call show()
   * the function uses a good 3d package that allows to draw and flowly navigate 
   into the 3D scene.
   * the cube object is drawin in red wirframe 
   * each camera is drawn as black pyramid 
     - its tip is on the camera location 
     - its base is a proportional to the camera resolution (1080x960) , but every 1000 pixels are scaled to 10cm 
     - the camera pyramid is oriented according to the camera direction 
     - a black 10 cm arrows is drawn from the camera tip to indicate the camera up direction 
     - a blue line is drown from the camera tip within its optical axq  is direction , the length of this line can bet by argument 
       and tis default is 10 meters 
   * the ground is drawn as light gray grid lines of 1 meter size 
   * background is white 
   * the naavigating of the 3d scene should be smooth and natural  e
     the user can zoom in and out using wheel, 
     the user can rotate with mouse movement to left and right ,
     the user can move in the scene using AWSDQE (W-forward, S-backward, A-left, D-right, Q-up, E-down)
    
"""

import random
import numpy as np
import pyvista as pv
from sim import sim_cam_m3, sim_math, sim_types


def create_cameras():
    """Create cam1 and cam2 with randomized parameters within spec ranges.

    Returns:
        tuple of ((cam1_intrinsics, cam1_transform), (cam2_intrinsics, cam2_transform))
    """
    z1 = random.uniform(1.3, 1.5)

    cam1 = sim_cam_m3.SimCamM3(width=1080, height=960, focal=1000)
    cam1_transform = sim_math.parameters_to_transform(
        sim_types.TransformParams(x=0, y=0, z=z1, roll=0, pitch=0, yaw=0))

    cam2 = sim_cam_m3.SimCamM3(width=1080, height=960, focal=1000)
    cam2_transform = sim_math.parameters_to_transform(
        sim_types.TransformParams(
            x=random.uniform(-0.3, 0.3),
            y=random.uniform(-1.0, -0.6),
            z=z1 + random.uniform(-0.3, 0.3),
            roll=0, pitch=0,
            yaw=random.uniform(-35, -25)))

    return (cam1, cam1_transform), (cam2, cam2_transform)


def create_cube():
    """Create a cube object with randomized dimensions within spec ranges.

    Returns:
        (center, L, W, H) where center is a 3D numpy array and L/W/H are floats in meters.
    """
    L = random.uniform(1.0, 3.0)
    W = random.uniform(0.2, 2.0)
    H = random.uniform(0.01, 2.0)
    dist = random.uniform(3.0, 5.0)
    center = np.array([dist, 0.0, H / 2.0])
    return center, L, W, H


def _draw_camera(plotter, transform, width, height, optical_axis_length):
    """Draw a camera in the 3D scene.

    The camera is represented as:
    - A black wireframe pyramid (tip at camera position, base proportional to resolution)
    - A black arrow indicating the camera up direction (10 cm)
    - A blue line along the optical axis
    """
    pos = transform[:3, 3].copy()
    fwd = transform[:3, 0].copy()
    right = transform[:3, 1].copy()
    up = transform[:3, 2].copy()

    # 1000 pixels = 0.1 m
    half_w = (width / 2.0) / 1000.0 * 0.1
    half_h = (height / 2.0) / 1000.0 * 0.1
    depth = half_w

    base_center = pos + depth * fwd
    c0 = base_center + half_w * right + half_h * up
    c1 = base_center - half_w * right + half_h * up
    c2 = base_center - half_w * right - half_h * up
    c3 = base_center + half_w * right - half_h * up

    pyramid_edges = np.array([
        pos, c0, pos, c1, pos, c2, pos, c3,
        c0, c1, c1, c2, c2, c3, c3, c0,
    ], dtype=np.float64)
    plotter.add_lines(pyramid_edges, color='black', width=2)

    up_arrow = pv.Arrow(start=tuple(pos), direction=tuple(up), scale=0.1,
                        tip_length=0.3, tip_radius=0.12, shaft_radius=0.04)
    plotter.add_mesh(up_arrow, color='black')

    axis_pts = np.array([pos, pos + optical_axis_length * fwd], dtype=np.float64)
    plotter.add_lines(axis_pts, color='blue', width=2)


def show(cam1_data, cam2_data, cube_center, cube_L, cube_W, cube_H,
         optical_axis_length=10.0):
    """Render the 3D scene with cameras, cube, and ground grid.

    Navigation:
        Mouse drag  - rotate the view
        Scroll      - zoom in/out
        W/S         - move forward/backward
        A/D         - move left/right
        Q/E         - move up/down
        Escape      - close the window
    """
    pl = pv.Plotter()
    pl.set_background('white')

    # --- ground grid (1 m spacing) ---
    grid_extent = 20
    grid_pts = []
    for i in range(-grid_extent, grid_extent + 1):
        grid_pts.extend([[float(i), float(-grid_extent), 0.0],
                         [float(i), float(grid_extent), 0.0]])
        grid_pts.extend([[float(-grid_extent), float(i), 0.0],
                         [float(grid_extent), float(i), 0.0]])
    pl.add_lines(np.array(grid_pts), color='lightgray', width=1)

    # --- cube (red wireframe) ---
    cube = pv.Cube(center=tuple(cube_center),
                   x_length=cube_L, y_length=cube_W, z_length=cube_H)
    pl.add_mesh(cube, style='wireframe', color='red', line_width=3)

    # --- cameras ---
    for cam_intrinsics, cam_transform in [cam1_data, cam2_data]:
        _draw_camera(pl, cam_transform, cam_intrinsics.width, cam_intrinsics.height,
                     optical_axis_length)

    # --- WASD + QE navigation ---
    # Disable the two sources of default key bindings, then register our own
    # movement keys through pyvista's key-event dispatch (which rides on the
    # KeyPressEvent observer pyvista sets up at construction time):
    #  1. VTK's interactor style maps single chars (w=wireframe, s=surface,
    #     e/q=exit, r=reset, ...) inside OnChar.  The style registers this as a
    #     'CharEvent' observer ON THE INTERACTOR, so we remove those observers.
    #     (Subclassing the style and overriding OnChar in Python does NOT work:
    #     VTK dispatches OnChar through an internal C++ callback that bypasses
    #     the Python override.)  Mouse interaction uses other events and stays.
    #  2. pyvista's own key callbacks (q=close, v, Up/Down, +/-, ...), stored in
    #     a python dict; cleared with clear_key_event_callbacks().
    pl.iren.interactor.RemoveObservers('CharEvent')
    pl.iren.clear_key_event_callbacks()

    move_speed = 0.3

    def _cam_vectors():
        p = np.array(pl.camera.position)
        f = np.array(pl.camera.focal_point)
        d = f - p
        d = d / np.linalg.norm(d)
        r = np.cross(d, np.array([0.0, 0.0, 1.0]))
        n = np.linalg.norm(r)
        if n > 1e-8:
            r = r / n
        else:
            r = np.array([0.0, 1.0, 0.0])
        return d, r

    def _move(delta):
        p = np.array(pl.camera.position)
        f = np.array(pl.camera.focal_point)
        pl.camera.position = (p + delta).tolist()
        pl.camera.focal_point = (f + delta).tolist()
        pl.render()

    def _forward():
        _move(_cam_vectors()[0] * move_speed)

    def _backward():
        _move(-_cam_vectors()[0] * move_speed)

    def _strafe_left():
        _move(-_cam_vectors()[1] * move_speed)

    def _strafe_right():
        _move(_cam_vectors()[1] * move_speed)

    def _up():
        _move(np.array([0.0, 0.0, move_speed]))

    def _down():
        _move(np.array([0.0, 0.0, -move_speed]))

    # register both lower- and upper-case so CapsLock/Shift still works
    for keys, cb in [(('w', 'W'), _forward),
                     (('s', 'S'), _backward),
                     (('a', 'A'), _strafe_left),
                     (('d', 'D'), _strafe_right),
                     (('q', 'Q'), _up),
                     (('e', 'E'), _down)]:
        for k in keys:
            pl.add_key_event(k, cb)

    # initial viewpoint: looking at the scene from behind-left and above
    pl.camera.position = (-5.0, -8.0, 6.0)
    pl.camera.focal_point = (4.0, 0.0, 0.5)
    pl.camera.up = (0.0, 0.0, 1.0)

    pl.show()


def main():
    random.seed(42)

    cam1_data, cam2_data = create_cameras()
    cube_center, cube_L, cube_W, cube_H = create_cube()

    cam1_intrinsics, cam1_transform = cam1_data
    cam2_intrinsics, cam2_transform = cam2_data

    print(f"cam1  pos={cam1_transform[:3, 3]}  fwd={cam1_transform[:3, 0]}")
    print(f"cam2  pos={cam2_transform[:3, 3]}  fwd={cam2_transform[:3, 0]}")
    print(f"Cube  center={cube_center}  L={cube_L:.2f}  W={cube_W:.2f}  H={cube_H:.2f}")

    show(cam1_data, cam2_data, cube_center, cube_L, cube_W, cube_H)


if __name__ == '__main__':
    main()
