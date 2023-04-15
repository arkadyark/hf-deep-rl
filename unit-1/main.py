"""
Implements unit 1 of the deep RL course
"""
from pathlib import Path

import gym
from huggingface_sb3 import load_from_hub, package_to_hub, push_to_hub
from pyvirtualdisplay import Display
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv


def env_info(env_name):
    """
    Print initial environment information
    """
    env = gym.make(env_name)
    _ = env.reset()
    for _ in range(20):
        # Take a random action
        action = env.action_space.sample()
        print("Action taken:", action)
        observation, reward, done, info = env.step(action)

        if done:
            observation = env.reset()
            print("Environment is reset")

    env = gym.make(env_name)
    print("___OBSERVATION SPACE___")
    print("Observation space shape:", env.observation_space.shape)
    print("Sample observation:", env.observation_space.sample())

    print("___ACTION SPACE___")
    print("Action space shape:", env.action_space.n)
    print("Action space sample:", env.action_space.sample())
    return env


def train(env_name, batch_size=16, num_steps=1000000, overwrite=False):
    """
    Train the model
    """
    # Vectorized environment, so we can batch examples during training
    model_name = f"ppo-{env_name}"
    if Path(model_name + ".zip").exists() and not overwrite:
        print(f"Loading pretrained model from {model_name}.zip")
        model = PPO.load(model_name)
    else:
        env = make_vec_env(env_name, n_envs=batch_size)
        model = PPO(
            policy="MlpPolicy",
            env=env,
            n_steps=1024,
            batch_size=64,
            n_epochs=4,
            gamma=0.999,
            gae_lambda=0.98,
            ent_coef=0.01,
            verbose=1,
        )
        model.learn(total_timesteps=num_steps)
        model.save(model_name)
    return model


def evaluate(env_name, model):
    """
    Evaluate the trained model
    """
    env = gym.make(env_name)
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=10, deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward}")


def publish(model, env_name, model_name):
    """
    Publish the model to the huggingface hub (requires login through the CLI)
    """
    repo_id = "arkadyark/deep-rl-course-unit-1"
    env_id = env_name
    eval_env = DummyVecEnv([lambda: gym.make(env_id)])
    model_architecture = "PPO"
    commit_message = "Push LunarLander-v2 model"

    package_to_hub(
        model,
        model_name=model_name,
        model_architecture=model_architecture,
        env_id=env_id,
        eval_env=eval_env,
        repo_id=repo_id,
        commit_message=commit_message,
    )


def main():
    """
    Train and evaluate our deep RL model for lunar policy
    """
    virtual_display = Display(visible=0, size=(1400, 900))
    virtual_display.start()
    env_name = "LunarLander-v2"
    model_name = f"arkadyark-{env_name}"
    overwrite = True
    env_info(env_name)
    model = train(env_name, overwrite=overwrite)
    publish(model, env_name, model_name)


if __name__ == "__main__":
    main()
