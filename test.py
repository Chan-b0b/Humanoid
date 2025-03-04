from stable_baselines3 import PPO
import gymnasium as gym
from test_env import UnitreeG1Env
import torch
import os
from model import CustomPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
import datetime
import numpy as np
from model import * 


# Register the environment
gym.register(
    id="UnitreeG1-v0",
    entry_point=UnitreeG1Env,
    max_episode_steps=1000,
)

if __name__ == "__main__":
    env = gym.make("UnitreeG1-v0", render_mode="human")

    policy_kwargs = dict(
        features_extractor_class=CNNExtractor,
        features_extractor_kwargs = dict(features_dim=256)
    )
    # env = Monitor(env)
    # envs = DummyVecEnv([lambda: Monitor(gym.make("UnitreeG1-v0", render_mode="human")) for _ in range(10)])
    envs = SubprocVecEnv([lambda: Monitor(gym.make("UnitreeG1-v0", render_mode="human")) for _ in range(10)])

    model = PPO(CustomPolicy, envs, policy_kwargs=policy_kwargs, batch_size=512, learning_rate = 1e-5, clip_range=0.1, verbose=1)
    # model = PPO.load('results/ppo_unitree_squat_0')
    # model.policy.load_state_dict(torch.load("/home/workspace/chlee/Humanoid/loco-mujoco/loco_mujoco/pre_results/pre_CNN_qvel_initial_xpos_30/ppo_unitree_stand_0.pth"))

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model.set_logger(configure(f'./logs/{current_time}', ['stdout','tensorboard']))

    test_name = 'pre_CNN_qvel_initial_xpos_30'
    save_dir = f'pre_results/{test_name}'

    # for sub_env in envs.envs:
    #     sub_env.logger = model.logger
        
    # Run simulation 
    for i in range(1000):
        model.learn(total_timesteps=100000, reset_num_timesteps=False)

        obs, _ = env.reset()
        done = False
        env.render()
        while not done:
            # action = env.action_space.sample()  # Random action
            action, _ = model.predict(obs)
            
            # action = np.array([ -5.81676149,   6.24023438,  -2.51686788, -10,   0.89736181, -1.86867142,   
            #           5.31338787,  -6.59179688,   3.69140625,  -7.47070312, 3.66783547,  -2.27591777,   
            #           0.16779119,   0.,           0.,
            #             0.625,        1.875,        0.5625,      -0.6875,       0.25,
            #             -0.234375,     0.14648438,   0.8125,      -2.125,       -0.5,
            #             -0.4375,      -0.1875,      -0.15625,     -0.15625,   ])/30
            
            obs, reward, done, truncated, info = env.step(action)
        
            env.render()  # Render frame
        
        if i % 5 == 0:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.policy.state_dict(), f'{save_dir}/ppo_unitree_stand_{i}.pth')
    env.close()  # Ensure cleanup