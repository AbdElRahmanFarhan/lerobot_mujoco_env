# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from loop_rate_limiters import RateLimiter
import numpy as np

import mujoco
import mujoco.viewer
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def inference():

    model = mujoco.MjModel.from_xml_path("/home/mink/examples/franka_emika_panda/scene.xml")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 480, 640)

    # policy_path = "/home/models/smol_vla_policy/panda_mujoco/reach_cube/panda_mujoco_lerobot/v2/"
    policy = SmolVLAPolicy.from_pretrained("Abderlrahman/smolvla-panda-mujoco-reach-cube")
    print(policy.config.image_features)


    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    # set random position for the box
    x_rand = np.random.uniform(low=-0.2, high=0.2) 
    y_rand = np.random.uniform(low=-0.3, high=0.3)
    x, y = data.qpos[9: 11]
    data.qpos[9: 11] = [x+x_rand, y+y_rand]
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=True, show_right_ui=True) as viewer:
                
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "camera_far")
        # mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        while viewer.is_running():

            # replace it with env
            observation = {}
            observation["agent_pos"] = np.array([data.qpos[i]*(180/np.pi) for i in range(7)])

            renderer.update_scene(data, camera="camera_hand")
            camera_hand_img = renderer.render()
            renderer.update_scene(data, camera="camera_far")
            camera_far_img = renderer.render()

            observation["pixels"] = {}
            observation["pixels"]["camera_hand"] = camera_hand_img
            observation["pixels"]["camera_far"] = camera_far_img

            observation = preprocess_observation(observation)

            # observation = {key: observation[key].to("cuda") for key in observation}

            observation["task"] = "reach cube"

            action = policy.select_action(observation)
            
            # Set robot controls (first 8 dofs in your configuration)
            # action = action.to("cpu").numpy()
            print(action)
            data.ctrl = action


            # Step simulation
            mujoco.mj_step(model, data)

            # rate.sleep()
            viewer.sync()



    

if __name__ == "__main__":
    inference()
