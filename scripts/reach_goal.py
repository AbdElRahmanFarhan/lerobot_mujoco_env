
import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink
import numpy as np

# IK parameters
SOLVER = "quadprog"
POS_THRESHOLD = 1e-4
ORI_THRESHOLD = 1e-4
MAX_ITERS = 20
NUM_EPISODES = 5
NUM_FRAMES = 300

def converge_ik(
    configuration, tasks, dt, solver, pos_threshold, ori_threshold, max_iters
):
    """
    Runs up to 'max_iters' of IK steps. Returns True if position and orientation
    are below thresholds, otherwise False.
    """
    for _ in range(max_iters):
        vel = mink.solve_ik(configuration, tasks, dt, solver, 1e-3)
        configuration.integrate_inplace(vel, dt)

        # Only checking the first FrameTask here (end_effector_task).
        # If you want to check multiple tasks, sum or combine their errors.
        err = tasks[0].compute_error(configuration)
        pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
        ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold

        if pos_achieved and ori_achieved:
            return True
    return False


def main():
    # Load model & data
    model = mujoco.MjModel.from_xml_path("/home/mink/examples/franka_emika_panda/scene.xml")
    data = mujoco.MjData(model)

    # Create a Mink configuration
    configuration = mink.Configuration(model)


    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "camera_hand")
    joint_names = [model.joint(i).name for i in range(9)]
    print(joint_names)

    # with mujoco.viewer.launch_passive(
    #     model=model, data=data, show_left_ui=True, show_right_ui=True
    # ) as viewer:
    #     # Set viewer to use the named camera
    #     viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    #     viewer.cam.fixedcamid = cam_id
    #     # mujoco.mjv_defaultFreeCamera(model, viewer.cam)

    #     for _ in range(NUM_EPISODES):
            
    #         # Reset simulation data to the 'home' keyframe
    #         mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)

    #         # set random position for the box
    #         x_rand = np.random.uniform(low=-0.1, high=0.1) 
    #         y_rand = np.random.uniform(low=-0.3, high=0.3)

    #         x, y = data.qpos[9: 11]
    #         data.qpos[9: 11] = [x+x_rand, y+y_rand]

    #         configuration.update(data.qpos)
    #         mujoco.mj_forward(model, data)

    #         # Define tasks
    #         posture_task = mink.PostureTask(model=model, cost=1e-2)
    #         posture_task.set_target_from_configuration(configuration)
            
    #         # We'll track time ourselves for a smoother trajectory
    #         local_time = 0.0
    #         rate = RateLimiter(frequency=50.0, warn=False)

    #         # Get box location
    #         box_pose = configuration.get_transform_frame_to_world("box", "body")
    #         box_translation = box_pose.translation()

    #         # Set a goal
    #         approach_translation = box_translation.copy()
    #         approach_translation[2] += 0.2
    #         ee_rotation = mink.SO3.from_rpy_radians(np.pi, 0.0, 0.0)

    #         approach_goal = mink.SE3.from_rotation_and_translation(ee_rotation, approach_translation)
    #         approach_task = mink.FrameTask(
    #             frame_name="attachment_site",
    #             frame_type="site",
    #             position_cost=1.0,
    #             orientation_cost=1.0,
    #             lm_damping=1.0,
    #         )
    #         approach_task.set_target(approach_goal)
    #         tasks = [approach_task, posture_task]
    #         success = False
    #         viewer.sync()
            
    #         for _ in range(NUM_FRAMES):
    #             # Update our local time
    #             dt = rate.dt
    #             local_time += dt
    #             converge_ik(
    #                 configuration,
    #                 tasks,
    #                 dt,
    #                 SOLVER,
    #                 POS_THRESHOLD,
    #                 ORI_THRESHOLD,
    #                 MAX_ITERS,
    #             )
    #             # Set robot controls (first 8 dofs in your configuration)
    #             data.ctrl = configuration.q[:8]
    #             # Step simulation
    #             mujoco.mj_step(model, data)

    #             frame_id = model.site("attachment_site").id 
    #             ee_translation = data.site_xpos[frame_id]

    #             if np.allclose(ee_translation, approach_translation, atol=1e-2):
    #                 success = True

    #             # Visualize at fixed FPS
    #             viewer.sync()
    #             rate.sleep()


if __name__ == "__main__":
    main()
