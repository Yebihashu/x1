from sim import ttl_testtools as ttl

#====================================================================
# Running tests using pytest
#====================================================================
import numpy as np
from grx import grx_lib, grx_math
from pathlib import Path

ENABLE_DEBUG = True

DEBUG_OUTPUT_FOLDER = Path(__file__).parent.parent.parent.parent.parent / "debug_grx"
if not DEBUG_OUTPUT_FOLDER.exists():
    DEBUG_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

def _debug_save_image(image, filename):
    if not ENABLE_DEBUG:
        return
    import PIL.Image
    image_np = np.asarray(image)
    if image_np.ndim == 4 and image_np.shape[0] == 1:
        image_np = image_np[0]
    image = PIL.Image.fromarray(image_np)
    path = (DEBUG_OUTPUT_FOLDER / f"{filename}.png").resolve()
    image.save(path)
    print(f"Saved image to: {path}")

def test_play__empty_scene__orthographic_camera():
    # 1. create ImagesBuffer of one frame rgb of size 200x100 pixels
    # 2. create orthographic camera of size of  width=200 pixels, and height - 100 pixels 
    #    position the camera at (0,1,0), where it look at (0,0,1) and up is (0,1,0)
    # 3. create viewer , no windows (headless mode) , background color is blue, and assign the camera to viewer     
    # 4. create a play object with update function that does the follows:
    #    for the first call - do nothing 
    #    for the second call (after first frame was rendered)  
    #        - copy image from viewer to buffer, image index 0. 
    #        - command to exit play
    # 5. Copy images from buffer to cpu numpy array, and verify the image in the buffer is blue  

    width, height = 200, 100

    # 1. images buffer: one rgb frame of 200x100
    images_buffer = grx_lib.ImagesBuffer(num_images=1, width=width, height=height, channels=3)

    camera_transform = grx_math.look_at_transform(position=np.array([0.0, 1.0, 0.0]),
                                                   look_at_position=np.array([0.0, 0.0, 1.0]),
                                                   strive_up=np.array([0.0, 1.0, 0.0]))

    camera = grx_lib.OrthographicCamera(name="cam", transform=camera_transform,
                                        width=width, height=height)

    # 3. headless viewer with a blue background and the camera assigned
    scene = grx_lib.Scene()
    view = grx_lib.Viewer(name="view", scene=scene, camera=camera, headless=True)
    view.set_background_color(grx_lib.BLUE)

    # 4. play object: do nothing on the first update, capture + exit on the second
    class Play(grx_lib.player):
        def __init__(self, viewers):
            super().__init__(viewers)
            self.update_calls = 0
            self.captured = False

        def _update(self):
            self.update_calls += 1
            if self.update_calls >= 2:
                # second call: the first frame has been rendered by now
                self.captured = self.get_viewer_image("view", images_buffer, 0)
                self.exit()

    play = Play([view])
    play.run()  # blocks until exit() stops the render loop

    assert play.captured, "failed to copy the viewer image into the buffer"

    # 5. copy images to a cpu numpy array and verify the frame is blue
    images = images_buffer.copy_to_cpu()
    _debug_save_image(images, "test_play__empty_scene")
    assert images.shape == (1, height, width, 3)
    image = images[0]
    assert np.all(image[:, :, 0] < 10), "red channel should be ~0 for a blue image"
    assert np.all(image[:, :, 1] < 10), "green channel should be ~0 for a blue image"
    assert np.all(image[:, :, 2] > 245), "blue channel should be ~255 for a blue image"

    

def test_play__cube__orthographic_camera():
    # 1. create ImagesBuffer of one frame rgb of size 200x100 pixels
    # 2. create orthographic camera of size of  width=200 pixels, and height - 100 pixels 
    #    position the camera at (0,100,0), where it look at (0,0,0) and up is (0,0,1)
    # 3. create white cube of size of (1,2,0.5) and place it at (-0.5,1,0.25)
    # 3. create viewer , no windows (headless mode) , background color is blue, and assign the camera to viewer     
    # 4. create a play object with update function that does the follows:
    #    for the first call - do nothing 
    #    for the second call (after first frame was rendered)  
    #        - copy image from viewer to buffer, image index 0. 
    #        - command to exit play
    # 5. Copy images from buffer to cpu numpy array, and verify the follow:
    #    - top-left area of image of 100x50 pixels is the cube. 
    #    - rest area of image is blue. 

    width, height = 200, 100

    # 1. images buffer: one rgb frame of 200x100
    images_buffer = grx_lib.ImagesBuffer(num_images=1, width=width, height=height, channels=3)

    # 2. orthographic top-down camera at (0,100,0) looking at the origin, up = +z.
    #    With the grx cs (x=right, y=up, z=front), the camera's right maps to world
    #    +x (image x) and its up maps to world +z (image y).
    camera_transform = grx_math.look_at_transform(position=np.array([0.0, 100.0, 0.0]),
                                                   look_at_position=np.array([0.0, 0.0, 0.0]),
                                                   strive_up=np.array([0.0, 0.0, 1.0]))
    camera = grx_lib.OrthographicCamera(name="cam", transform=camera_transform,
                                        width=width, height=height)

    # 3. white cube of size (x=1, y=2, z=0.5) centered at (-0.5, 1, 0.25).
    #    default ortho film is 2.0m x 1.0m (100 px/m), centered on the optical axis,
    #    so the cube's front face (x in [-1,0], z in [0,0.5]) projects to the
    #    top-left 100x50 pixels of the image.
    scene = grx_lib.Scene()
    cube = grx_lib.Cube(name="cube", size=[1.0, 2.0, 0.5], color=[1.0, 1.0, 1.0, 1.0])
    scene.spawn_object(cube, transform=grx_lib.translation_transform(-0.5, 1.0, 0.25))

    # 3. headless viewer, blue background, lights OFF (default) -> flat cube color.
    view = grx_lib.Viewer(name="view", scene=scene, camera=camera, headless=True)
    view.set_background_color(grx_lib.BLUE)

    # 4. play object: do nothing on the first update, capture + exit on the second
    class Play(grx_lib.player):
        def __init__(self, viewers):
            super().__init__(viewers)
            self.update_calls = 0
            self.captured = False

        def _update(self):
            self.update_calls += 1
            if self.update_calls >= 2:
                self.captured = self.get_viewer_image("view", images_buffer, 0)
                self.exit()

    play = Play([view])
    play.run()  # blocks until exit() stops the render loop

    assert play.captured, "failed to copy the viewer image into the buffer"

    # 5. copy images to a cpu numpy array and verify the layout
    images = images_buffer.copy_to_cpu()
    _debug_save_image(images, "test_play__cube__orthographic_camera")
    assert images.shape == (1, height, width, 3)
    image = images[0]

    def is_white(region):
        return np.all(region[:, :, 0] > 245) and np.all(region[:, :, 1] > 245) and np.all(region[:, :, 2] > 245)

    def is_blue(region):
        return (np.all(region[:, :, 0] < 10) and np.all(region[:, :, 1] < 10)
                and np.all(region[:, :, 2] > 245))

    # top-left 100x50 is the white cube (checked on an inner core to avoid the
    # exact 1-pixel boundary where anti-aliasing may blend cube and background).
    assert is_white(image[0:50, 0:100]), "top-left area should be the white cube"

    # the rest of the image should be blue
    assert is_blue(image[50:100, :]), "bottom half should be blue background"
    assert is_blue(image[:, 100:200]), "right half should be blue background"


#====================================================================
# Running tests maunally
#====================================================================
if __name__ == "__main__":
    # Run TestSystem tests
    ttl.run_tests_manually([test_play__cube__orthographic_camera])
