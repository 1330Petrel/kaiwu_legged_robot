# pyright: reportAttributeAccessIssue=false
# pyright: reportMissingImports=false

import time
import mujoco
from threading import Thread
import threading
import glfw
from typing import Optional, Any

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand

import config


locker = threading.Lock()

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)

elastic_band: Optional[Any] = None
band_attached_link: Optional[int] = None
if config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()
    if config.ROBOT == "h1" or config.ROBOT == "g1":
        band_attached_link = mj_model.body("torso_link").id
    else:
        band_attached_link = mj_model.body("base_link").id

# 自定义 scene 和 context，使用更大的 maxgeom
MAX_GEOM = 100000

# 初始化 GLFW
if not glfw.init():
    raise Exception("Could not initialize GLFW")

# 创建窗口
window = glfw.create_window(1280, 720, "Unitree MuJoCo Simulator", None, None)
if not window:
    glfw.terminate()
    raise Exception("Could not create GLFW window")

glfw.make_context_current(window)
glfw.swap_interval(1)

# 创建 MuJoCo 可视化对象
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
scene = mujoco.MjvScene(mj_model, maxgeom=MAX_GEOM)
context = mujoco.MjrContext(mj_model, mujoco.mjtFontScale.mjFONTSCALE_150)

# 初始化相机
cam.azimuth = 90
cam.elevation = -20
cam.distance = 3.0
cam.lookat[:] = [0, 0, 0.5]

# 鼠标控制变量
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

def mouse_button_callback(window, button, act, mods):
    global button_left, button_middle, button_right
    button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

def mouse_move_callback(window, xpos, ypos):
    global lastx, lasty, button_left, button_middle, button_right
    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    if not (button_left or button_middle or button_right):
        return

    width, height = glfw.get_window_size(window)
    mod_shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
                 glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)

    if button_right:
        action = mujoco.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mujoco.mjtMouse.mjMOUSE_ZOOM

    mujoco.mjv_moveCamera(mj_model, action, dx/width, dy/height, scene, cam)

def scroll_callback(window, xoffset, yoffset):
    mujoco.mjv_moveCamera(mj_model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05*yoffset, scene, cam)

# 设置回调
glfw.set_mouse_button_callback(window, mouse_button_callback)
glfw.set_cursor_pos_callback(window, mouse_move_callback)
glfw.set_scroll_callback(window, scroll_callback)

class ViewerHandle:
    def __init__(self, window):
        self._window = window
        self._running = True
    
    def is_running(self):
        return self._running and not glfw.window_should_close(self._window)
    
    def sync(self):
        global scene, context, cam, opt
        mujoco.mjv_updateScene(mj_model, mj_data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        
        viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(self._window))
        mujoco.mjr_render(viewport, scene, context)
        glfw.swap_buffers(self._window)
        glfw.poll_events()
    
    def close(self):
        self._running = False
        glfw.terminate()

viewer = ViewerHandle(window)

mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)


def SimulationThread():
    global mj_data, mj_model

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)

    if config.USE_JOYSTICK:
        unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()

        if config.ENABLE_ELASTIC_BAND and elastic_band is not None and band_attached_link is not None:
            if elastic_band.enable:
                mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )
        mujoco.mj_step(mj_model, mj_data)

        locker.release()

        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


def PhysicsViewerThread():
    while viewer.is_running():
        viewer.sync()
        time.sleep(config.VIEWER_DT)


if __name__ == "__main__":
    sim_thread = Thread(target=SimulationThread)
    sim_thread.start()
    
    PhysicsViewerThread()
