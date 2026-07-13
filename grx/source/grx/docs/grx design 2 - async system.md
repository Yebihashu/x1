THIS CODUMENT IS A FUTURE GRX SYSTEM DESIGN - DO NOT USE IT FOR CURRENT CODE

# GRX lib - general
1. IMPORTANT: GRX is intended to be a real-time 3D engine, so it must be fast, efficient, and responsive.
2. GRX is currently based on the Panda3D rendering engine. Panda3D-specific code should be isolated inside the render backend layer, so the rendering backend can be changed later if needed.
3. The GRX coordinate system is defined in "\grx\source\grx\docs\coordinates system.md".
4. Units are meters and degrees.
5. Colors are RGBA values in the range 0.0-1.0. Example: [1.0, 0.0, 0.0, 1.0] is opaque red.
6. Transparency is supported. It should be possible to enable or disable transparency per object and/or per scene/viewer.
7. GRX uses NumPy for efficient math calculations. The math backend should be reasonably isolated so it can be replaced or accelerated later if needed.
8. Transforms are 4x4 matrices using column-vector convention. Transforming a homogeneous point is done as:
   transformed_point = transform_matrix @ vector_point
9. GRX should use a non-blocking log system. Log calls must not delay rendering, user input, or scene updates. Log calls should enqueue messages quickly, and a separate logger thread should handle printing/writing messages. The logger can print to stdout, file, or both. The default is stdout only.
   Note: the logger may use its own queue. This is separate from the render synchronization design. GRX should avoid a per-object render command queue for normal scene updates.
10. GRX should allow fluid navigation and rendering, even with two viewers or more, and should respond quickly to keyboard/mouse input. Heavy user update logic must not block rendering. Therefore, GRX should use the parallel update/render architecture described below.


# Parallel Update/Render Scene Synchronization Design
The following architecture must satisfy the general constraints above.



## Concept

The GRX runtime should separate the **logical scene** from the **rendered scene**.

The update/application thread runs the user logic and owns the real, correct scene state. This is the scene used by `_update()`, `set_transform()`, `get_transform()`, object creation/removal, physics, AI, camera logic, and any other computation that needs the current status of scene objects.

The rendering thread owns only the visual representation of the scene. It draws the latest scene state it has received, but it is allowed to be slightly behind the update/application thread. This visual delay is usually only a fraction of a second, often one frame, and is not noticeable to the user.

NOTE: The update/application thread may be the Python main thread. However, the design should not depend on this, because some 3D APIs require rendering to run on the Python main thread. The important rule is ownership: the update/application thread owns the logical scene, and the render thread owns Panda3D visual objects.



This gives the engine two important performance advantages:

1. The heavy `_update()` logic can run in parallel with rendering.
2. The update/application thread does not need to lock the scene or enqueue every small operation as a command.

Instead of sharing one mutable scene between both threads, the system uses **two different responsibilities**:

```text
Update/application thread:
    owns the true logical scene
    freely modifies scene objects
    answers get_transform() and other logic queries
    publishes frame diffs or frame state to renderer

Render thread:
    owns Panda3D objects / NodePaths
    receives published changes
    updates its visual cache
    renders the scene
```

This follows the original GRX principle that Panda3D visual objects should only be touched by the rendering thread, while NumPy transforms and scene logic remain the authoritative model.

## Why no locks are needed for scene updates

There is no conflict because both threads do not mutate the same scene data.

The update/application thread owns the logical scene. It can freely call:

```python
obj.set_transform(...)
obj.get_transform(...)
scene.spawn_object(...)
scene.remove_object(...)
viewer.set_background_color(...)
```

without locking, because the render thread does not read or mutate that same Python scene object.

The render thread owns its own visual scene. It owns:

```text
Panda3D NodePaths
render cameras
render lights
render-side viewer state
render-side object cache
```

The render thread never asks the logical scene for current values while the update/application thread may be modifying it. Instead, it receives completed published updates.

The synchronization boundary is not the scene object itself. The synchronization boundary is a **published frame packet**.

```text
_update() modifies logical scene
_update() finishes
update/application thread publishes FramePacket
render thread consumes latest FramePacket
render thread applies it to Panda3D
render thread draws
```

Because the renderer consumes only completed packets, it never sees a half-updated scene.

## Why no command queue is needed

The design avoids a command queue like:

```text
set_transform() -> enqueue command
set_color()     -> enqueue command
set_visible()   -> enqueue command
spawn_object()  -> enqueue command
remove_object() -> enqueue command
```

That kind of queue creates many tiny Python objects and many queue operations, especially when thousands of objects are updated every frame.

Instead, `_update()` changes the logical scene directly. At the end of the update step, GRX publishes a compact description of what changed.

That published data can be called:

```text
FramePacket
SceneDiff
RenderUpdate
PublishedFrameState
```

The important idea is that it is **batched**. The render thread receives one completed update packet, not thousands of tiny commands.

## What data is sent to the render thread

The render thread needs only render-relevant changes.

The update/application thread does not copy the entire Python scene graph. It sends compact diffs such as:

```text
changed transforms
changed visibility
changed colors
changed materials
added objects
removed objects
reparented objects
camera changes
light changes
viewer changes
```

A typical packet may look conceptually like this:

```python
FramePacket:
    frame_id
    spawned_objects
    removed_object_ids
    reparented_objects

    dirty_transform_ids
    transforms

    dirty_visibility_ids
    visibility

    dirty_color_ids
    colors

    dirty_material_ids
    materials

    dirty_camera_ids
    camera_states

    viewer_changes
```

For high performance, frequently changing data should be stored in flat arrays rather than copied as Python objects:

```python
local_transforms: np.ndarray  # shape: (N, 4, 4)
visible: np.ndarray           # shape: (N,)
colors: np.ndarray            # shape: (N, 4)
materials: array/list/table
```

Then the update/application thread can publish only the dirty object IDs and the matching values:

```python
dirty_transform_ids = [4, 18, 91]
transforms = local_transforms[dirty_transform_ids]
```

This is much faster than deep-copying scene objects.

## Object identity

Each scene object should have a stable integer object ID.

Names are useful for user lookup:

```text
"ego_car" -> object_id 17
```

But the render thread should mostly work with IDs:

```text
object_id 17 -> Panda3D NodePath
object_id 17 -> latest transform row
object_id 17 -> material
```

This avoids expensive string lookups during rendering.

A strong implementation can use an ID plus generation number:

```text
ObjectHandle = (object_id, generation)
```

That prevents stale updates from accidentally affecting a newly created object that reused the same ID.

## Transform updates

When `_update()` calls:

```python
car.set_transform(new_transform)
```

the logical scene immediately updates the car’s transform. Any later call to:

```python
car.get_transform()
```

inside the update logic returns the latest logical value.

The object ID is marked dirty:

```text
dirty_transform_ids.add(car.id)
```

At the end of `_update()`, the frame packet includes the dirty transform IDs and their transforms.

The render thread receives the packet and applies them to the render objects:

```text
for object_id in dirty_transform_ids:
    nodepath = render_nodepaths[object_id]
    nodepath.setMat(grx_to_panda(transform[object_id]))
```

The current GRX code already has conversion helpers between GRX’s left-handed `x=right, y=up, z=front` transform convention and Panda3D’s matrix convention, so the render thread should continue using that conversion path.

## Color and material updates

Colors and materials should follow the same dirty-state idea.

When the update/application thread changes an object color:

```python
cube.set_color([1.0, 0.0, 0.0, 1.0])
```

the logical object is updated immediately and the object ID is marked dirty:

```text
dirty_color_ids.add(cube.id)
```

The frame packet contains:

```text
dirty_color_ids
new_colors
```

The render thread applies those changes to the matching Panda3D visual object.

For materials, the packet may contain material IDs or material descriptors:

```text
object_id
material_id
material_properties
```

Examples of material state:

```text
base color
alpha/transparency
roughness
metallic
texture reference
shader type
lighting enabled/disabled
```

For performance, materials should ideally be shared by ID, not copied per object every frame.

## Visibility updates

Visibility is also dynamic render state.

The update/application thread can freely call:

```python
obj.set_visible(False)
```

The logical scene marks the object visibility dirty.

The render thread later applies:

```text
visible=True  -> nodepath.show()
visible=False -> nodepath.hide()
```

This is important for object pooling, temporary debug objects, and fast hide/show behavior.

## Adding objects while running

Adding an object is a structural change, not just a state change.

When the update/application thread calls:

```python
scene.spawn_object(Cube("box"))
```

the logical scene immediately:

```text
allocates a stable object ID
adds the object to the name registry
sets its parent
sets its initial transform
sets geometry/material metadata
marks it as alive
```

Then the next frame packet includes a spawn diff:

```text
SpawnObject:
    object_id
    name
    object_type
    parent_id
    initial_transform
    geometry parameters
    material/color
    visibility
```

The render thread consumes the spawn diff and creates the Panda3D visual object on the render thread.

For example:

```text
Cube -> create box geometry NodePath
Arrow -> create line/arrow NodePath
Axes -> create axis NodePath
Camera -> create/update camera representation
Light -> create Panda3D light
```

Then it stores:

```text
render_nodepaths[object_id] = nodepath
```

and applies the initial transform, material, and visibility.

This preserves the important rule: the update/application thread owns the logical object, but only the render thread creates Panda3D objects.

## Removing objects while running

Removing an object is also a structural change.

When the update/application thread calls:

```python
scene.remove_object("box")
```

the logical scene immediately removes or marks the object as removed. If the object has children, the logical scene computes the full subtree to remove.

The next frame packet includes:

```text
RemoveObjects:
    object_ids = [parent_id, child_id_1, child_id_2, ...]
```

The render thread consumes this and calls remove/destroy on the matching Panda3D nodes.

For frequent add/remove patterns, the render thread should prefer pooling:

```text
instead of destroying NodePath:
    hide it
    detach or move to pool
    reuse it for a later object of the same type
```

This is especially useful for debug markers, particles, arrows, temporary boxes, or frequently changing annotations.

## Reparenting objects

Changing parent is also structural.

The update scene immediately updates:

```text
object.parent_id
parent.children
local/world transform relationship
```

The frame packet sends:

```text
ReparentObject:
    object_id
    new_parent_id
    local_transform_after_reparent
```

The render thread applies:

```text
nodepath.reparentTo(parent_nodepath)
nodepath.setMat(local_transform)
```

This keeps the visual hierarchy aligned with the logical hierarchy.

## Camera updates

Cameras are scene objects too, so their transforms follow the same transform-diff system.

If the update/application thread changes a camera transform, the camera ID is marked dirty and sent to the renderer.

The render thread then updates the Panda3D camera transform before rendering.

Camera intrinsic changes, such as focal length, resolution, orthographic film size, or projection mode, should be sent as camera property diffs:

```text
CameraLensChanged:
    camera_id
    projection_type
    focal
    ppx
    ppy
    width
    height
    film_width
    film_height
```

The current GRX design already separates pinhole and orthographic camera concepts, including pinhole intrinsics and orthographic film size.

## Light updates

Lights should use the same split.

Logical light state belongs to the logical scene:

```text
light transform
light color
light intensity
light enabled/disabled
```

The render thread owns the actual Panda3D light object.

Light changes are sent as diffs:

```text
LightChanged:
    light_id
    transform
    color
    intensity
    enabled
```

The render thread applies them before rendering.

## Viewer behavior changes

Viewer changes are not object transforms, but they are still render-side changes.

Examples:

```python
viewer.set_background_color(...)
viewer.set_grid_visible(True)
viewer.set_grid(...)
viewer.enable_lights(True)
viewer.enable(False)
```

The update/application thread should update the logical viewer state immediately, then publish viewer diffs.

For example:

```text
ViewerBackgroundChanged:
    viewer_id
    color
```

The render thread applies:

```text
graphics_output.setClearColor(color)
```

For grid changes:

```text
ViewerGridChanged:
    viewer_id
    visible
    cell_size
    color
```

The render thread creates, updates, shows, or hides the grid NodePath.

For lighting mode:

```text
ViewerLightingChanged:
    viewer_id
    enable_lights
```

The render thread applies the lighting state to that viewer’s camera/render state.

For enabling/disabling a viewer:

```text
ViewerEnabledChanged:
    viewer_id
    enabled
```

The render thread shows/hides the relevant render root or render output.

This matches the existing GRX viewer concept where background, grid, lighting, and enabled state belong to `Viewer`, while the actual visual application is done in the rendering engine.

## Render thread frame flow

The render thread loop becomes:

```text
while running:
    take latest published frame packet, if available

    apply structural removals
    apply structural spawns
    apply reparenting

    apply transform changes
    apply visibility changes
    apply color/material changes
    apply camera changes
    apply light changes
    apply viewer changes

    render frame
```

It is acceptable if the renderer skips old packets and uses only the latest complete one.

This is important. If `_update()` runs faster than rendering, the renderer should not build a backlog of old frames. It should prefer latest-state semantics:

```text
render newest available complete scene state
discard stale intermediate visual states
```

That keeps latency low.

## Update/application thread frame flow

The update/application thread loop becomes:

```text
while running:
    begin update frame

    _update()
        freely modifies logical scene
        uses get_transform() on logical scene
        adds/removes objects
        changes viewer state
        marks dirty state

    build/publish FramePacket

    continue
```

The update/application thread does not wait for the renderer except possibly during shutdown or when explicitly requesting a synchronized snapshot.

## How the visual scene follows the logical scene

The render scene follows the logical scene by applying every published diff.

For normal frame-to-frame motion, this means transform diffs.

For appearance changes, this means color/material/visibility diffs.

For object lifetime changes, this means spawn/remove diffs.

For hierarchy changes, this means reparent diffs.

For viewer changes, this means viewer diffs.

The renderer is therefore always converging toward the logical scene, but it does not need to be perfectly simultaneous with it. The logical scene is the truth. The rendered scene is a visual cache of the latest published truth.

## Final summary

The method is:

```text
Use the update/application thread for the real scene and heavy update logic.
Use the render thread for drawing only.
Do not share mutable scene objects between the threads.
Do not lock every object update.
Do not enqueue every setter call.
Instead, publish compact frame diffs from the update scene to the render scene.
The render scene may lag slightly, but remains visually smooth and safe.
```

This gives GRX a high-performance architecture:

```text
correctness:
    all logic reads the true update-side scene

performance:
    update and rendering run in parallel
    no per-object lock overhead
    no per-setter command queue overhead

safety:
    Panda3D remains render-thread-only
    render thread never reads half-mutated logical scene

visual quality:
    renderer draws the latest complete published state
    one-frame delay is usually invisible to the user
```
