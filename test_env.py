import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from mujoco import viewer
from dm_control import mjcf
from mushroom_rl.utils.mujoco import ObservationType
from stable_baselines3.common.logger import Logger
from collections import deque
import torch
from copy import deepcopy

class UnitreeG1Env(gym.Env):
    """
    Custom Gym environment for the Unitree G1 robot using MuJoCo.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode="human"):
        super(UnitreeG1Env, self).__init__()
        
        # Load MuJoCo model
        # model_path = "/home/workspace/chlee/Humanoid/loco-mujoco/loco_mujoco/environments/data/unitree_g1/g1.xml"
        model_path = "/home/workspace/chlee/Humanoid/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"

        self.model = mujoco.MjModel.from_xml_path(model_path)  # Ensure the XML file is correctly defined
        # self.model.qpos0[7:] = np.array([-0.26775169, -0.02886438,  0.02316952,  0.51035547, -0.2644988,   0.02171616,
        #                         -0.23803723,  0.00667286, -0.00588036,  0.45417786, -0.25589615, -0.05182436,
        #                         -0.00171757,  0.,          0.,          0.2911973,   0.21535492, -0.02535069,
        #                         0.97964668,  0.06205654,  0.07514071, -0.0302031,   0.29245567, -0.21655273,
        #                         0.03467464,  0.97902369, -0.12390709,  0.04775953,  0.04608782])


        self.data = mujoco.MjData(self.model)

        self._xml_handle = mjcf.from_path(model_path)

        self.action_spec = self._get_action_specification()
        self.observation_spec = self._get_observation_specification()

        """
        00 =  'left_hip_pitch_joint'
        01 =  'left_hip_roll_joint'
        02 =  'left_hip_yaw_joint'
        03 =  'left_knee_joint'
        04 =  'left_ankle_pitch_joint'
        05 =  'left_ankle_roll_joint'
        06 =  'right_hip_pitch_joint'
        07 =  'right_hip_roll_joint'
        08 =  'right_hip_yaw_joint'
        09 =  'right_knee_joint'
        10 =  'right_ankle_pitch_joint'
        11 =  'right_ankle_roll_joint'
        12 =  'waist_yaw'
        13 =  'left_shoulder_pitch_joint'
        14 =  'left_shoulder_roll_joint'
        15 =  'left_shoulder_yaw_joint'
        16 =  'left_elbow_pitch_joint'
        17 =  'left_wrist_roll_joint'
        18 =  'right_shoulder_pitch_joint'
        19 =  'right_shoulder_roll_joint'
        20 =  'right_shoulder_yaw_joint'
        21 =  'right_elbow_pitch_joint'
        22 =  'right_wrist_roll_joint'
        """

        collision_groups = [("floor", ["floor"]),
                            ("right_foot_1", ["right_foot_1_col"]),
                            ("right_foot_2", ["right_foot_2_col"]),
                            ("right_foot_3", ["right_foot_3_col"]),
                            ("right_foot_4", ["right_foot_4_col"]),
                            ("left_foot_1", ["left_foot_1_col"]),
                            ("left_foot_2", ["left_foot_2_col"]),
                            ("left_foot_3", ["left_foot_3_col"]),
                            ("left_foot_4", ["left_foot_4_col"])]
        
        self.render_mode = render_mode
        self.viewer = None
        self.logger = None  # SB3 logger (will be set later)


        
        self.frame_skip = 2  # Simulation steps per action


        self.frame_stack = 15
        self.observation_dim = 100

        # Action space (torques applied to joints)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(29,), dtype=np.float64 
        )
        
        # Observation space (joint positions, velocities, and base state)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.frame_stack*self.observation_dim,), dtype=np.float64
        )

        self.obs_buf = deque(np.zeros((self.frame_stack, self.observation_dim)), maxlen=self.frame_stack)
    
    def step(self, action):
        """Apply action and update the simulation."""
        action = np.clip(action, self.action_space.low, self.action_space.high) * 20
        # action = np.array([ -5.81676149,   6.24023438,  -2.51686788, -10,   0.89736181,
        #             -1.86867142,   5.31338787,  -6.59179688,   3.69140625,  -7.47070312,
        #             3.66783547,  -2.27591777,   0.16779119,   0.,           0.,
        #             0.625,        1.875,        0.5625,      -0.6875,       0.25,
        #             -0.234375,     0.14648438,   0.8125,      -2.125,       -0.5,
        #             -0.4375,      -0.1875,      -0.15625,     -0.15625   ])

        self.data.ctrl[:] = action  # Apply torques
        # self.data.qpos[7:] = action
        # self.data.qpos[[12,13,14]] = 0
        # self.data.qvel[[12,13,14]] = 0

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

            # Get observations
            observation = self._get_obs()
            done = self._is_done()
            if done:
                break
    

        reward = self._compute_reward()

        info = {}
        if done:
            print(f"Episode finished in {self.data.time:.2f} seconds")
            # if self.logger:
            #     self.logger.record("episode/episode_time", self.data.time)
            info['episode_time'] = self.data.time

        self.obs_buf.append(observation)

        return np.array(self.obs_buf).flatten(), reward, done, False, info
    
    def reset(self, seed=None, options=None):
        """Reset the environment to its initial state."""
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        #차렷
        self.data.qpos[7:] = np.array([-0.26775169, -0.02886438,  0.02316952,  0.51035547, -0.2644988,   0.02171616,
                                 -0.23803723,  0.00667286, -0.00588036,  0.45417786, -0.25589615, -0.05182436,
                                 -0.00171757,  0.,          0.,          0.2911973,   0.21535492, -0.02535069,
                                 0.97964668,  0.06205654,  0.07514071, -0.0302031,   0.29245567, -0.21655273,
                                 0.03467464,  0.97902369, -0.12390709,  0.04775953,  0.04608782])
        self.data.qvel[:] = 0

        self.fallen = False
        self.pre_fall = None
        self.pre_pos = None
        self.target_height = self._get_obs()[2]
        self.init_pos = deepcopy(self.data.xpos[:])


        self.obs_buf = deque(np.zeros((self.frame_stack, self.observation_dim)), maxlen=self.frame_stack)
        
        return np.array(self.obs_buf).flatten(), {}
    
    def render(self):
        """Render the simulation."""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = viewer.launch_passive(self.model, self.data)
            else:
                self.viewer.sync()
        elif self.render_mode == "rgb_array":
            return self._get_camera_image()
    
    def close(self):
        """Close the environment and viewer."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    
    def _get_obs(self):
        """Retrieve the current observation from the simulation."""
        return np.concatenate([self.data.qpos[:], self.data.qvel[:], self.data.ctrl[:]])
    
    def _is_done(self):
        """Check if termination conditions are met."""
        if self._is_fallen():  # If the robot falls below a threshold
            return True
        if self.data.time > 10:
            return True
        return False
    
    def _is_fallen(self):
        # if self.data.qpos[2]< 0.20 :
        #     self.fallen = True
        
        rot_mat = np.zeros((3, 3)).flatten()
        mujoco.mju_quat2Mat(rot_mat, self.data.xquat[1])
        rot_mat = rot_mat.reshape(3,3)

        roll_angle = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
        pitch_angle = np.arctan2(-rot_mat[2, 0], np.sqrt(rot_mat[2, 1]**2 + rot_mat[2, 2]**2))

        if abs(pitch_angle) >= 1.8 or abs(roll_angle) >= 1.5:
            self.fallen = True

        return self.fallen
        # if abs(self.data.qpos[0]) > 0.3 or abs(self.data.qpos[1]) > 0.3 :
        # if abs(self.data.qpos[1]) > 0.3 :
        #     self.fallen = True

    def _get_camera_image(self):
        """Return an RGB image from a virtual camera."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        mujoco.mjr_render(640, 480, self.viewer.cam, img)
        return img

    def _compute_reward(self):
        """Reward function for squatting behavior."""
        base_height = self.data.qpos[2]  # z-position of the base
        left_opposites = [7, 11, 16, 20, 23]
        right_opposites = [13, 17, 23, 25, 28]
        left_equal = [6,8,9,10,19,21,22]
        right_equal = [12,14,15,16,24,26,27]

        # Encourage squatting motion (target height = 0.3 meters)
        squat_target = 0
        height_reward = -abs(base_height - squat_target)  # Max reward at target height

        # Penalize excessive joint velocity
        smooth_motion_penalty = -np.sum(np.abs(self.data.qvel)) * 0.1  

        # Encourage stability (don't fall)
        # stability_bonus = 1.0 if base_height > -0.4 else -1.0  # Reward staying upright

        # opposites_symmetry = -np.sum(np.abs(self.data.qpos[left_opposites] + self.data.qpos[right_opposites]))
        # equal_symmetry = -np.sum(np.abs(self.data.qpos[left_equal] - self.data.qpos[right_equal]))

        # symmetry_reward = opposites_symmetry + equal_symmetry

        joint_vel_penalty = -np.mean(np.abs(self.data.qvel[:]))  
    
        if self.fallen:
            stability_bonus = -10.0
        else:
            stability_bonus = 0

        foot_force = sum(np.linalg.norm(self.data.cfrc_ext[foot_id, :3]) for foot_id in [7,13])
        ground_contact_reward = np.tanh(foot_force / 100.0)  # Normalize force reward


        # initial_reward = -abs(self._get_obs()[:17]).sum()
        # current_fall = -abs(self.target_height - self.data.qpos[2])
        # current_fall_reward = current_fall - self.pre_fall

        # self.pre_fall = current_fall

        fall = abs(self.data.qpos[2]) - abs(self.pre_fall) if self.pre_fall is not None else 0
        self.pre_fall = self.data.qpos[2]

        rot_mat = np.zeros((3, 3)).flatten()
        mujoco.mju_quat2Mat(rot_mat, self.data.xquat[1])
        rot_mat = rot_mat.reshape(3,3)

        roll_angle = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
        pitch_angle = np.arctan2(-rot_mat[2, 0], np.sqrt(rot_mat[2, 1]**2 + rot_mat[2, 2]**2))
        tilt_penalty = -abs(roll_angle) - abs(pitch_angle)

        # return height_reward + velocity_penalty + stability_bonus

        if self.pre_pos is None:
            pos_penalty = 0
        else:
            pos_penalty = (abs(self.init_pos - self.pre_pos) - abs(self.init_pos - self.data.xpos)).mean()
        
        self.pre_pos = deepcopy(self.data.xpos)

        # return fall + tilt_penalty + pos_penalty
        return pos_penalty + joint_vel_penalty*0.2
        
    def _get_action_specification(self):
        """
        Getter for the action space specification.

        Returns:
            A list of tuples containing the specification of each action
            space entry.

        """

        action_spec = []
        actuators = self._xml_handle.find_all("actuator")
        for actuator in actuators:
            action_spec.append(actuator.name)
        return action_spec
    
    def _get_observation_specification(self):
        """
        Getter for the observation space specification.

        Returns:
            A list of tuples containing the specification of each observation
            space entry.

        """

        observation_spec = []
        for prefix in ["q_", "dq_"]:
            for j in self._xml_handle.find_all("joint"):
                obs_type = ObservationType.JOINT_POS if prefix == "q_" else ObservationType.JOINT_VEL
                observation_spec.append((prefix + j.name, j.name, obs_type))
        return observation_spec