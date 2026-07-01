from sim import ttl_testtools as ttl

#====================================================================
# Running tests using pytest
#====================================================================
import numpy as np
from grx import grx_lib


def test_play__empty_scene():
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

    # 2. orthographic camera looking from (0,1,0) toward (0,0,1), up (0,1,0)
    camera_transform = grx_lib.camera_look_at_transform(position=[0.0, 1.0, 0.0],
                                                        look_at=[0.0, 0.0, 1.0],
                                                        up=[0.0, 1.0, 0.0])
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
    assert images.shape == (1, height, width, 3)
    image = images[0]
    assert np.all(image[:, :, 0] < 10), "red channel should be ~0 for a blue image"
    assert np.all(image[:, :, 1] < 10), "green channel should be ~0 for a blue image"
    assert np.all(image[:, :, 2] > 245), "blue channel should be ~255 for a blue image"






#====================================================================
# Running tests maunally
#====================================================================
if __name__ == "__main__":
    # Run TestSystem tests
    ttl.run_tests_manually()
