import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from loop_rate_limiters import RateLimiter

import numpy as np
import rerun as rr

from lerobot.common.datasets.image_writer import safe_stop_image_writer
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import build_dataset_frame, hw_to_dataset_features
import mujoco
import mujoco.viewer
import mink

from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
from lerobot.common.utils.visualization_utils import _init_rerun
from lerobot.configs import parser


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

@dataclass
class DatasetRecordConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str = "Abderlrahman/panda_mujoco_lerobot"
    # A short but accurate description of the task performed during the recording (e.g. "Pick the Lego block and drop it in the box on the right.")
    single_task: str = "reach cube"
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path  = "/home/test"
    # Limit the frames per second.
    fps: int = 30
    # Number of seconds for data recording for each episode.
    episode_time_s: int | float = 8
    # Number of seconds for resetting the environment after each episode.
    reset_time_s: int | float = 60
    # Number of episodes to record.
    num_episodes: int = 50
    # Encode frames in the dataset into video
    video: bool = True
    # Upload dataset to Hugging Face hub.
    push_to_hub: bool = False
    # Upload on private repository on the Hugging Face hub.
    private: bool = False
    # Add tags to your dataset on the hub.
    tags: list[str] | None = None
    # Number of subprocesses handling the saving of frames as PNG. Set to 0 to use threads only;
    # set to ≥1 to use subprocesses, each using threads to write images. The best number of processes
    # and threads depends on your system. We recommend 4 threads per camera with 0 processes.
    # If fps is unstable, adjust the thread count. If still unstable, try using 1 or more subprocesses.
    num_image_writer_processes: int = 0
    # Number of threads writing the frames as png images on disk, per camera.
    # Too many threads might cause unstable teleoperation fps due to main thread being blocked.
    # Not enough threads might cause low camera fps.
    num_image_writer_threads_per_camera: int = 1

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")


@dataclass
class RecordConfig:
    dataset: DatasetRecordConfig
    # Display all cameras on screen
    display_data: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Resume recording on an existing dataset.
    resume: bool = False


@safe_stop_image_writer
def record_loop(
    dataset: LeRobotDataset,
    data: None,
    model: None,
    configuration: None,
    renderer: None,
    single_task: str,
    control_time_s: int,
    display_data: bool = False,
):

    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)

    # set random position for the box
    x_rand = np.random.uniform(low=-0.2, high=0.2) 
    y_rand = np.random.uniform(low=-0.3, high=0.3)

    x, y = data.qpos[9: 11]
    data.qpos[9: 11] = [x+x_rand, y+y_rand]

    configuration.update(data.qpos)
    mujoco.mj_forward(model, data)

    # Define tasks
    posture_task = mink.PostureTask(model=model, cost=1e-2)
    posture_task.set_target_from_configuration(configuration)
    
    # We'll track time ourselves for a smoother trajectory
    rate = RateLimiter(frequency=30.0, warn=False)

    # Get box location
    box_pose = configuration.get_transform_frame_to_world("box", "body")
    box_translation = box_pose.translation()

    # Set a goal
    approach_translation = box_translation.copy()
    approach_translation[2] += 0.2
    ee_rotation = mink.SO3.from_rpy_radians(np.pi, 0.0, 0.0)

    approach_goal = mink.SE3.from_rotation_and_translation(ee_rotation, approach_translation)
    approach_task = mink.FrameTask(
        frame_name="attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    approach_task.set_target(approach_goal)
    tasks = [approach_task, posture_task]


    timestamp = 0.0

    while timestamp < control_time_s:
        dt = rate.dt
        timestamp += dt

        # replace it with env
        observation = {f"{model.joint(i).name}.pos":data.qpos[i]*(180/np.pi) for i in range(7)}

        renderer.update_scene(data, camera="camera_hand")
        camera_hand_img = renderer.render()
        observation["camera_hand"] = camera_hand_img

        renderer.update_scene(data, camera="camera_far")
        camera_far_img = renderer.render()
        observation["camera_far"] = camera_far_img

        observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")

        converge_ik(configuration,
            tasks,
            dt,
            SOLVER,
            POS_THRESHOLD,
            ORI_THRESHOLD,
            MAX_ITERS,)
        
        # Set robot controls (first 8 dofs in your configuration)
        data.ctrl = configuration.q[:8] 
        action = {f"{model.joint(i).name}.pos":configuration.q[i]*(180/np.pi) for i in range(7)}
        action_frame = build_dataset_frame(dataset.features, action, prefix="action")

        # Step simulation
        mujoco.mj_step(model, data)

        frame = {**observation_frame, **action_frame}
        dataset.add_frame(frame, task=single_task)

        rate.sleep()

@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    model = mujoco.MjModel.from_xml_path("/home/mink/examples/franka_emika_panda/scene.xml")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 480, 640)
    cameras = ["camera_hand", "camera_far"]
    # Create a Mink configuration
    configuration = mink.Configuration(model)

    joint_names = [model.joint(i).name for i in range(7)]
    joint_features = {f"{joint_names[i]}.pos": float for i in range(7)}
    camera_features = {"camera_hand": (480, 640, 3),
                       "camera_far": (480, 640, 3)}

    observation_features = {**joint_features, **camera_features}
    action_features = joint_features

    act_features = hw_to_dataset_features(action_features, "action", cfg.dataset.video)
    obs_features = hw_to_dataset_features(observation_features, "observation", cfg.dataset.video)
    dataset_features = {**act_features, **obs_features}

    dataset = LeRobotDataset.create(
        cfg.dataset.repo_id,
        cfg.dataset.fps,
        root=cfg.dataset.root,
        features=dataset_features,
        use_videos=cfg.dataset.video,
        image_writer_processes=cfg.dataset.num_image_writer_processes,
        image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera*len(cameras),
    )

    for recorded_episodes in range(cfg.dataset.num_episodes):
        print(f"Recording episode {dataset.num_episodes}")
        record_loop(
            dataset=dataset,
            data=data,
            model=model,
            configuration=configuration,
            renderer=renderer,
            control_time_s=cfg.dataset.episode_time_s,
            single_task=cfg.dataset.single_task,
            display_data=cfg.display_data,
        )
        dataset.save_episode()

    print("Stop recording")

    if cfg.dataset.push_to_hub:
        dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    return dataset


if __name__ == "__main__":
    dataset_config = DatasetRecordConfig()
    record_config = RecordConfig(dataset=dataset_config)
    record(record_config)
