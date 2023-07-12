from pathlib import Path
from copy import deepcopy
import numpy as np
from dm_control import mjcf

from mushroom_rl.utils.running_stats import *
from mushroom_rl.utils.mujoco import *

from loco_mujoco.environments import LocoEnv
from loco_mujoco.utils import check_validity_task_mode_dataset

VALID_TASKS = ["walk", "carry"]
VALID_DATASET_TYPES = ["real", "perfect"]


class Atlas(LocoEnv):
    """
    Mujoco simulation of the Atlas robot. Optionally, Atlas can carry
    a weight. This environment can be partially observable by hiding
    some of the state space entries from the policy using a state mask.
    Hidable entries are "positions", "velocities", "foot_forces",
    or "weight".

    """

    def __init__(self, disable_arms=False, hold_weight=False, weight_mass=None, tmp_dir_name=None, **kwargs):
        """
        Constructor.

        """

        if hold_weight:
            assert disable_arms is True, "If you want Atlas to carry a weight, please disable the arms. " \
                                         "They will be kept fixed."

        xml_path = (Path(__file__).resolve().parent.parent / "data" / "atlas" / "model.xml").as_posix()

        action_spec = self._get_action_specification()

        observation_spec = self._get_observation_specification()

        collision_groups = [("floor", ["ground"]),
                            ("foot_r", ["right_foot_back"]),
                            ("front_foot_r", ["right_foot_front"]),
                            ("foot_l", ["left_foot_back"]),
                            ("front_foot_l", ["left_foot_front"])]

        self._hidable_obs = ("positions", "velocities", "foot_forces", "weight")

        # --- Modify the xml, the action_spec, and the observation_spec if needed ---
        self._disable_arms = disable_arms
        self._hold_weight = hold_weight
        self._weight_mass = weight_mass
        self._valid_weights = [0.1, 1.0, 5.0, 10.0]

        if disable_arms or hold_weight:
            xml_handle = mjcf.from_path(xml_path)

            if disable_arms:
                joints_to_remove, motors_to_remove, equ_constr_to_remove = self._get_xml_modifications()
                obs_to_remove = ["q_" + j for j in joints_to_remove] + ["dq_" + j for j in joints_to_remove]
                observation_spec = [elem for elem in observation_spec if elem[0] not in obs_to_remove]
                action_spec = [ac for ac in action_spec if ac not in motors_to_remove]

                xml_handle = self._delete_from_xml_handle(xml_handle, joints_to_remove,
                                                          motors_to_remove, equ_constr_to_remove)

            xml_path = []
            if hold_weight and weight_mass is not None:
                color_red = np.array([1.0, 0.0, 0.0, 1.0])
                xml_handle = self._add_weight(xml_handle, weight_mass, color_red)
                xml_path.append(self._save_xml_handle(xml_handle, tmp_dir_name))
            elif hold_weight and weight_mass is None:
                for i, w in enumerate(self._valid_weights):
                    color = self._get_box_color(i)
                    current_xml_handle = deepcopy(xml_handle)
                    current_xml_handle = self._add_weight(current_xml_handle, w, color)
                    xml_path.append(self._save_xml_handle(current_xml_handle, tmp_dir_name))
            else:
                xml_path.append(self._save_xml_handle(xml_handle, tmp_dir_name))

        super().__init__(xml_path, action_spec, observation_spec, collision_groups, **kwargs)

    # def setup(self, obs):
    #     """
    #     Function to setup the initial state of the simulation. Initialization can be done either
    #     randomly, from a certain initial, or from the default initial state of the model.
    #
    #     Args:
    #         obs (np.array): Observation to initialize the environment from;
    #
    #     """
    #
    #     self._reward_function.reset_state()
    #
    #     super().setup(obs)
    #
    #     if self._hold_weight:
    #         if self._weight_mass is None:
    #             ind = np.random.randint(0, len(self._valid_weights))
    #             new_weight_mass = self._valid_weights[ind]
    #             self._model.body("weight").mass = new_weight_mass
    #
    #             # todo: also change the inertial of the mass
    #
    #             # modify the color of the mass according to the mass
    #             red_rgba = np.array([[1.0, 0.0, 0.0, 1.0]])
    #             blue_rgba = np.array([[0.2, 0.0, 1.0, 1.0]])
    #             interpolation_var = ind / (len(self._valid_weights)-1)
    #             color = blue_rgba + ((red_rgba - blue_rgba) * interpolation_var)
    #             geom_id = self._model.body("weight").geomadr[0]
    #             self._model.geom_rgba[geom_id] = color
    #         else:
    #             self._model.body("weight").mass = self._weight_mass

    def create_dataset(self, ignore_keys=None):
        """
        Creates a dataset from the specified trajectories.

        Args:
            ignore_keys (list): List of keys to ignore in the dataset. Default is ["q_pelvis_tx", "q_pelvis_tz"].

        Returns:
            Dictionary containing states, next_states and absorbing flags. For the states the shape is
            (N_traj x N_samples_per_traj, dim_state), while the absorbing flag has the shape is
            (N_traj x N_samples_per_traj).

        """

        if ignore_keys is None:
            ignore_keys = ["q_pelvis_tx", "q_pelvis_tz"]

        dataset = super().create_dataset(ignore_keys)

        return dataset

    def get_mask(self, obs_to_hide):
        """
        This function returns a boolean mask to hide observations from a fully observable state.

        Args:
            obs_to_hide (tuple): A tuple of strings with names of objects to hide.
            Hidable objects are "positions", "velocities", "foot_forces", and "env_type".

        Returns:
            Mask in form of a np.array of booleans. True means that that the obs should be
            included, and False means that it should be discarded.

        """

        if type(obs_to_hide) == str:
            obs_to_hide = (obs_to_hide,)
        assert all(x in self._hidable_obs for x in obs_to_hide), "Some of the observations you want to hide are not" \
                                                                 "supported. Valid observations to hide are %s." \
                                                                 % (self._hidable_obs,)

        pos_dim, vel_dim = self._len_qpos_qvel()
        force_dim = self._get_grf_size()

        mask = []
        if "positions" not in obs_to_hide:
            mask += [np.ones(pos_dim, dtype=np.bool)]
        else:
            mask += [np.zeros(pos_dim, dtype=np.bool)]

        if "velocities" not in obs_to_hide:
            mask += [np.ones(vel_dim, dtype=np.bool)]
        else:
            mask += [np.zeros(vel_dim, dtype=np.bool)]

        if self._use_foot_forces:
            if "foot_forces" not in obs_to_hide:
                mask += [np.ones(force_dim, dtype=np.bool)]
            else:
                mask += [np.zeros(force_dim, dtype=np.bool)]
        else:
            assert "foot_forces" not in obs_to_hide, "Creating a mask to hide foot forces without activating " \
                                                     "the latter is not allowed."

        if self._hold_weight:
            if "weight" not in obs_to_hide:
                mask += [np.ones(1, dtype=np.bool)]
            else:
                mask += [np.zeros(1, dtype=np.bool)]
        else:
            assert "weight" not in obs_to_hide, "Creating a mask to hide the carried weight without activating " \
                                                "the latter is not allowed."

        return np.concatenate(mask).ravel()

    def _get_xml_modifications(self):
        """
        Function that specifies which joints, motors and equality constraints
        should be removed from the Mujoco xml.

        Returns:
            A tuple of lists consisting of names of joints to remove, names of motors to remove,
             and names of equality constraints to remove.

        """

        joints_to_remove = []
        motors_to_remove = []
        equ_constr_to_remove = []

        if self._disable_arms:
            joints_to_remove += ["l_arm_shz", "l_arm_shx", "l_arm_ely", "l_arm_elx", "l_arm_wry", "l_arm_wrx",
                                 "r_arm_shz", "r_arm_shx", "r_arm_ely", "r_arm_elx", "r_arm_wry", "r_arm_wrx"]
            motors_to_remove += ["l_arm_shz_actuator", "l_arm_shx_actuator", "l_arm_ely_actuator", "l_arm_elx_actuator",
                                 "l_arm_wry_actuator", "l_arm_wrx_actuator", "r_arm_shz_actuator", "r_arm_shx_actuator",
                                 "r_arm_ely_actuator", "r_arm_elx_actuator", "r_arm_wry_actuator", "r_arm_wrx_actuator"]

        return joints_to_remove, motors_to_remove, equ_constr_to_remove

    def _get_observation_space(self):
        """
        Returns a tuple of the lows and highs (np.array) of the observation space.

        """

        low, high = super(Atlas, self)._get_observation_space()
        if self._hold_weight:
            low = np.concatenate([low, [self._valid_weights[0]]])
            high = np.concatenate([high, [self._valid_weights[-1]]])

        return low, high

    def _create_observation(self, obs):
        """
        Creates a full vector of observations.

        Args:
            obs (np.array): Observation vector to be modified or extended;
            return_err_msg (bool): If True, an error message with violations is returned.

        Returns:
            New observation vector (np.array).

        """

        obs = super(Atlas, self)._create_observation(obs)
        if self._hold_weight:
            weight_mass = deepcopy(self._model.body("weight").mass)
            obs = np.concatenate([obs, weight_mass])

        return obs

    def _has_fallen(self, obs, return_err_msg=False):
        """
        Checks if a model has fallen.

        Args:
            obs (np.array): Current observation.
            return_err_msg (bool): If True, an error message with violations is returned.

        Returns:
            True, if the model has fallen for the current observation, False otherwise.
            Optionally an error message is returned.

        """

        pelvis_euler = self._get_from_obs(obs, ["q_pelvis_tilt", "q_pelvis_list", "q_pelvis_rotation"])
        pelvis_y_condition = (obs[0] < -0.3) or (obs[0] > 0.1)
        pelvis_tilt_condition = (pelvis_euler[0] < (-np.pi / 4.5)) or (pelvis_euler[0] > (np.pi / 12))
        pelvis_list_condition = (pelvis_euler[1] < -np.pi / 12) or (pelvis_euler[1] > np.pi / 8)
        pelvis_rotation_condition = (pelvis_euler[2] < (-np.pi / 10)) or (pelvis_euler[2] > (np.pi / 10))
        pelvis_condition = (pelvis_y_condition or pelvis_tilt_condition or
                            pelvis_list_condition or pelvis_rotation_condition)

        back_euler = self._get_from_obs(obs, ["q_back_bky", "q_back_bkx", "q_back_bkz"])

        back_extension_condition = (back_euler[0] < (-np.pi / 4)) or (back_euler[0] > (np.pi / 10))
        back_bending_condition = (back_euler[1] < -np.pi / 10) or (back_euler[1] > np.pi / 10)
        back_rotation_condition = (back_euler[2] < (-np.pi / 4.5)) or (back_euler[2] > (np.pi / 4.5))
        back_condition = (back_extension_condition or back_bending_condition or back_rotation_condition)

        if return_err_msg:
            error_msg = ""
            if pelvis_y_condition:
                error_msg += "pelvis_y_condition violated.\n"
            elif pelvis_tilt_condition:
                error_msg += "pelvis_tilt_condition violated.\n"
            elif pelvis_list_condition:
                error_msg += "pelvis_list_condition violated.\n"
            elif pelvis_rotation_condition:
                error_msg += "pelvis_rotation_condition violated.\n"
            elif back_extension_condition:
                error_msg += "back_extension_condition violated.\n"
            elif back_bending_condition:
                error_msg += "back_bending_condition violated.\n"
            elif back_rotation_condition:
                error_msg += "back_rotation_condition violated.\n"

            return pelvis_condition or back_condition, error_msg
        else:

            return pelvis_condition or back_condition

    def _get_box_color(self, ind):
        """
        Calculates the rgba color based on the index of the environment.

        Args:
            ind (int): Current index of the environment.

        Returns:
            rgba np.array.

        """

        red_rgba = np.array([1.0, 0.0, 0.0, 1.0])
        blue_rgba = np.array([0.2, 0.0, 1.0, 1.0])
        interpolation_var = ind / (len(self._valid_weights) - 1)
        color = blue_rgba + ((red_rgba - blue_rgba) * interpolation_var)

        return color

    @staticmethod
    def generate(task="walk", dataset_type="real", gamma=0.99, horizon=1000, disable_arms=True,
                 use_foot_forces=False, random_start=True, init_step_no=None):
        """
        Returns an Atlas environment corresponding to the specified task.

        Args:
            task (str): Main task to solve. Either "walk" or "carry". The latter is walking while carrying
                an unknown weight, which makes the task partially observable.
            dataset_type (str): "real" or "perfect". "real" uses real motion capture data as the
                reference trajectory. This data does not perfectly match the kinematics
                and dynamics of this environment, hence it is more challenging. "perfect" uses
                a perfect dataset.
            gamma (float): Discounting parameter of the environment.
            horizon (int): Horizon of the environment.
            disable_arms (bool): If True, arms are disabled.
            use_foot_forces (bool): If True, foot forces are added to the observation space.
            random_start (bool): If True, a random sample from the trajectories
                is chosen at the beginning of each time step and initializes the
                simulation according to that.
            init_step_no (int): If set, the respective sample from the trajectories
                is taken to initialize the simulation.

        Returns:
            An MDP of the Atlas Robot.

        """
        check_validity_task_mode_dataset(Atlas.__name__, task, None, dataset_type,
                                         VALID_TASKS, None, VALID_DATASET_TYPES)


        # Generate the MDP
        if task == "walk":
            mdp = Atlas(gamma=gamma, horizon=horizon, random_start=random_start, init_step_no=init_step_no,
                        disable_arms=disable_arms, use_foot_forces=use_foot_forces)
        elif task == "carry":
            mdp = Atlas(gamma=gamma, horizon=horizon, random_start=random_start, init_step_no=init_step_no,
                        disable_arms=disable_arms, use_foot_forces=use_foot_forces, hold_weight=True)

        # Load the trajectory
        env_freq = 1 / mdp._timestep  # hz
        desired_contr_freq = 1 / mdp.dt  # hz
        n_substeps = env_freq // desired_contr_freq

        if dataset_type == "real":
            traj_data_freq = 500  # hz
            traj_params = dict(traj_path="../datasets/humanoids/02-constspeed_ATLAS.npz",
                               traj_dt=(1 / traj_data_freq),
                               control_dt=(1 / desired_contr_freq))
        elif dataset_type == "perfect":
            # todo: generate and add this dataset
            raise ValueError(f"currently not implemented.")

        mdp.load_trajectory(traj_params, warn=False)

        return mdp

    @staticmethod
    def _add_weight(xml_handle, mass, color):
        """
        Adds a weight to the Mujoco XML handle. The weight will
        be hold in front of Atlas. Therefore, the arms will be
        reoriented.

        Args:
            xml_handle: Handle to Mujoco XML.

        Returns:
            Modified Mujoco XML handle.

        """

        # find pelvis handle
        pelvis = xml_handle.find("body", "utorso")
        pelvis.add("body", name="weight")
        weight = xml_handle.find("body", "weight")
        weight.add("geom", type="box", size="0.1 0.27 0.1", pos="0.72 0 -0.25", rgba=color, mass=mass)

        # modify the arm orientation
        r_clav = xml_handle.find("body", "r_clav")
        r_clav.quat = [1.0,  0.0, -0.35, 0.0]
        l_clav = xml_handle.find("body", "l_clav")
        l_clav.quat = [0.0, -0.35, 0.0,  1.0]

        return xml_handle

    @staticmethod
    def _get_observation_specification():
        """
        Getter for the observation space specification.

        Returns:
            A list of tuples containing the specification of each observation
            space entry.

        """

        observation_spec = [# ------------- JOINT POS -------------
                            ("q_pelvis_tx", "pelvis_tx", ObservationType.JOINT_POS),
                            ("q_pelvis_tz", "pelvis_tz", ObservationType.JOINT_POS),
                            ("q_pelvis_ty", "pelvis_ty", ObservationType.JOINT_POS),
                            ("q_pelvis_tilt", "pelvis_tilt", ObservationType.JOINT_POS),
                            ("q_pelvis_list", "pelvis_list", ObservationType.JOINT_POS),
                            ("q_pelvis_rotation", "pelvis_rotation", ObservationType.JOINT_POS),
                            ("q_back_bkz", "back_bkz", ObservationType.JOINT_POS),
                            ("q_back_bkx", "back_bkx", ObservationType.JOINT_POS),
                            ("q_back_bky", "back_bky", ObservationType.JOINT_POS),
                            ("q_l_arm_shz", "l_arm_shz", ObservationType.JOINT_POS),
                            ("q_l_arm_shx", "l_arm_shx", ObservationType.JOINT_POS),
                            ("q_l_arm_ely", "l_arm_ely", ObservationType.JOINT_POS),
                            ("q_l_arm_elx", "l_arm_elx", ObservationType.JOINT_POS),
                            ("q_l_arm_wry", "l_arm_wry", ObservationType.JOINT_POS),
                            ("q_l_arm_wrx", "l_arm_wrx", ObservationType.JOINT_POS),
                            ("q_r_arm_shz", "r_arm_shz", ObservationType.JOINT_POS),
                            ("q_r_arm_shx", "r_arm_shx", ObservationType.JOINT_POS),
                            ("q_r_arm_ely", "r_arm_ely", ObservationType.JOINT_POS),
                            ("q_r_arm_elx", "r_arm_elx", ObservationType.JOINT_POS),
                            ("q_r_arm_wry", "r_arm_wry", ObservationType.JOINT_POS),
                            ("q_r_arm_wrx", "r_arm_wrx", ObservationType.JOINT_POS),
                            ("q_hip_flexion_r", "hip_flexion_r", ObservationType.JOINT_POS),
                            ("q_hip_adduction_r", "hip_adduction_r", ObservationType.JOINT_POS),
                            ("q_hip_rotation_r", "hip_rotation_r", ObservationType.JOINT_POS),
                            ("q_knee_angle_r", "knee_angle_r", ObservationType.JOINT_POS),
                            ("q_ankle_angle_r", "ankle_angle_r", ObservationType.JOINT_POS),
                            ("q_hip_flexion_l", "hip_flexion_l", ObservationType.JOINT_POS),
                            ("q_hip_adduction_l", "hip_adduction_l", ObservationType.JOINT_POS),
                            ("q_hip_rotation_l", "hip_rotation_l", ObservationType.JOINT_POS),
                            ("q_knee_angle_l", "knee_angle_l", ObservationType.JOINT_POS),
                            ("q_ankle_angle_l", "ankle_angle_l", ObservationType.JOINT_POS),

                            # ------------- JOINT VEL -------------
                            ("dq_pelvis_tx", "pelvis_tx", ObservationType.JOINT_VEL),
                            ("dq_pelvis_tz", "pelvis_tz", ObservationType.JOINT_VEL),
                            ("dq_pelvis_ty", "pelvis_ty", ObservationType.JOINT_VEL),
                            ("dq_pelvis_tilt", "pelvis_tilt", ObservationType.JOINT_VEL),
                            ("dq_pelvis_list", "pelvis_list", ObservationType.JOINT_VEL),
                            ("dq_pelvis_rotation", "pelvis_rotation", ObservationType.JOINT_VEL),
                            ("dq_back_bkz", "back_bkz", ObservationType.JOINT_VEL),
                            ("dq_back_bkx", "back_bkx", ObservationType.JOINT_VEL),
                            ("dq_back_bky", "back_bky", ObservationType.JOINT_VEL),
                            ("dq_l_arm_shz", "l_arm_shz", ObservationType.JOINT_VEL),
                            ("dq_l_arm_shx", "l_arm_shx", ObservationType.JOINT_VEL),
                            ("dq_l_arm_ely", "l_arm_ely", ObservationType.JOINT_VEL),
                            ("dq_l_arm_elx", "l_arm_elx", ObservationType.JOINT_VEL),
                            ("dq_l_arm_wry", "l_arm_wry", ObservationType.JOINT_VEL),
                            ("dq_l_arm_wrx", "l_arm_wrx", ObservationType.JOINT_VEL),
                            ("dq_r_arm_shz", "r_arm_shz", ObservationType.JOINT_VEL),
                            ("dq_r_arm_shx", "r_arm_shx", ObservationType.JOINT_VEL),
                            ("dq_r_arm_ely", "r_arm_ely", ObservationType.JOINT_VEL),
                            ("dq_r_arm_elx", "r_arm_elx", ObservationType.JOINT_VEL),
                            ("dq_r_arm_wry", "r_arm_wry", ObservationType.JOINT_VEL),
                            ("dq_r_arm_wrx", "r_arm_wrx", ObservationType.JOINT_VEL),
                            ("dq_hip_flexion_r", "hip_flexion_r", ObservationType.JOINT_VEL),
                            ("dq_hip_adduction_r", "hip_adduction_r", ObservationType.JOINT_VEL),
                            ("dq_hip_rotation_r", "hip_rotation_r", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_r", "knee_angle_r", ObservationType.JOINT_VEL),
                            ("dq_ankle_angle_r", "ankle_angle_r", ObservationType.JOINT_VEL),
                            ("dq_hip_flexion_l", "hip_flexion_l", ObservationType.JOINT_VEL),
                            ("dq_hip_adduction_l", "hip_adduction_l", ObservationType.JOINT_VEL),
                            ("dq_hip_rotation_l", "hip_rotation_l", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_l", "knee_angle_l", ObservationType.JOINT_VEL),
                            ("dq_ankle_angle_l", "ankle_angle_l", ObservationType.JOINT_VEL)]

        return observation_spec

    @staticmethod
    def _get_action_specification():
        """
        Getter for the action space specification.

        Returns:
            A list of tuples containing the specification of each action
            space entry.

        """

        action_spec = ["back_bkz_actuator", "back_bky_actuator", "back_bkx_actuator", "l_arm_shz_actuator",
                       "l_arm_shx_actuator", "l_arm_ely_actuator", "l_arm_elx_actuator", "l_arm_wry_actuator",
                       "l_arm_wrx_actuator", "r_arm_shz_actuator", "r_arm_shx_actuator",
                       "r_arm_ely_actuator", "r_arm_elx_actuator", "r_arm_wry_actuator", "r_arm_wrx_actuator",
                       "hip_flexion_r_actuator", "hip_adduction_r_actuator", "hip_rotation_r_actuator",
                       "knee_angle_r_actuator", "ankle_angle_r_actuator", "hip_flexion_l_actuator",
                       "hip_adduction_l_actuator", "hip_rotation_l_actuator", "knee_angle_l_actuator",
                       "ankle_angle_l_actuator"]

        return action_spec
