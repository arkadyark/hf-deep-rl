import datetime
import json
import os
import pickle
import random
from pathlib import Path

import gym
import imageio
import numpy as np
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.repocard import metadata_eval_result, metadata_save
from pyvirtualdisplay import Display
from tqdm.notebook import tqdm


def init_env(env_id):
    """ """
    env = gym.make(env_id)

    print("___OBSERVATION SPACE___")
    print("Observation space shape:", env.observation_space)
    print("Sample observation:", env.observation_space.sample())

    print("___ACTION SPACE___")
    print("Action space shape:", env.action_space.n)
    print("Action space sample:", env.action_space.sample())

    return env


def train(
    env,
    n_training_episodes=1000000,
    learning_rate=0.7,
    max_steps=99,
    gamma=0.95,
    max_epsilon=1.0,
    min_epsilon=0.05,
    decay_rate=0.0005,
):
    # Initialize Q table
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    epsilon = max_epsilon
    for episode in range(n_training_episodes):
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
            -decay_rate * episode
        )
        state, done = env.reset(), False
        for step in range(max_steps):
            action = epsilon_greedy_policy(q_table, state, epsilon)
            new_state, reward, done, _ = env.step(action)
            q_table[state, action] += learning_rate * (
                reward
                + gamma * q_table[new_state, greedy_policy(q_table, new_state)]
                - q_table[state, action]
            )
            state = new_state
            if done:
                break
    return q_table


def greedy_policy(q_table, state):
    action = q_table[state].argmax()
    return action


def epsilon_greedy_policy(q_table, state, epsilon):
    if random.random() < epsilon:
        action = random.randrange(0, q_table.shape[1])
    else:
        action = greedy_policy(q_table, state)
    return action


def eval_agent(model, env, max_steps=99, n_eval_episodes=100, seed=list()):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param Q: The Q-table
    :param seed: The evaluation seed array (for taxi-v3)
    """
    episode_rewards = []
    for episode in tqdm(range(n_eval_episodes)):
        if seed:
            state = env.reset(seed=seed[episode])
        else:
            state = env.reset()
        step = 0
        done = False
        total_rewards_ep = 0

        for step in range(max_steps):
            # Take the action (index) that have the maximum expected future reward given that state
            action = greedy_policy(model, state)
            new_state, reward, done, _ = env.step(action)
            total_rewards_ep += reward

            if done:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


def record_video(env, Qtable, out_directory, fps=1):
    """
    Generate a replay video of the agent
    :param env
    :param Qtable: Qtable of our agent
    :param out_directory
    :param fps: how many frame per seconds (with taxi-v3 and frozenlake-v1 we use 1)
    """
    images = []
    done = False
    state = env.reset(seed=random.randint(0, 500))
    img = env.render(mode="rgb_array")
    images.append(img)
    while not done:
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(Qtable[state][:])
        state, reward, done, info = env.step(
            action
        )  # We directly put next_state = state for recording logic
        img = env.render(mode="rgb_array")
        images.append(img)
    imageio.mimsave(
        out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps
    )

def push_to_hub(repo_id, model, env, video_fps=1, local_repo_path="hub"):
    """
    Evaluate, Generate a video and Upload a model to Hugging Face Hub.
    This method does the complete pipeline:
    - It evaluates the model
    - It generates the model card
    - It generates a replay video of the agent
    - It pushes everything to the Hub

    :param repo_id: repo_id: id of the model repository from the Hugging Face Hub
    :param env
    :param video_fps: how many frame per seconds to record our video replay
    (with taxi-v3 and frozenlake-v1 we use 1)
    :param local_repo_path: where the local repository is
    """
    _, repo_name = repo_id.split("/")

    eval_env = env
    api = HfApi()

    # Step 1: Create the repo
    repo_url = api.create_repo(
        repo_id=repo_id,
        exist_ok=True,
    )

    # Step 2: Download files
    repo_local_path = Path(snapshot_download(repo_id=repo_id))

    # Step 3: Save the model
    if env.spec.kwargs.get("map_name"):
        model["map_name"] = env.spec.kwargs.get("map_name")
        if env.spec.kwargs.get("is_slippery", "") == False:
            model["slippery"] = False

    # Pickle the model
    with open((repo_local_path) / "q-learning.pkl", "wb") as f:
        pickle.dump(model, f)

    # Step 4: Evaluate the model and build JSON with evaluation metrics
    mean_reward, std_reward = eval_agent(
        model["qtable"], eval_env, max_steps=model["max_steps"], n_eval_episodes=model["n_eval_episodes"], seed=model["eval_seed"]
    )

    evaluate_data = {
        "env_id": model["env_id"],
        "mean_reward": mean_reward,
        "n_eval_episodes": model["n_eval_episodes"],
        "eval_datetime": datetime.datetime.now().isoformat(),
    }

    # Write a JSON file called "results.json" that will contain the
    # evaluation results
    with open(repo_local_path / "results.json", "w") as outfile:
        json.dump(evaluate_data, outfile)

    # Step 5: Create the model card
    env_name = model["env_id"]
    if env.spec.kwargs.get("map_name"):
        env_name += "-" + env.spec.kwargs.get("map_name")

    if env.spec.kwargs.get("is_slippery", "") == False:
        env_name += "-" + "no_slippery"

    metadata = {}
    metadata["tags"] = [env_name, "q-learning", "reinforcement-learning", "custom-implementation"]

    # Add metrics
    eval = metadata_eval_result(
        model_pretty_name=repo_name,
        task_pretty_name="reinforcement-learning",
        task_id="reinforcement-learning",
        metrics_pretty_name="mean_reward",
        metrics_id="mean_reward",
        metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
        dataset_pretty_name=env_name,
        dataset_id=env_name,
    )

    # Merges both dictionaries
    metadata = {**metadata, **eval}

    model_card = f"""
    # **Q-Learning** Agent playing1 **{env_name}**
    This is a trained model of a **Q-Learning** agent playing **{env_name}** .

    ## Usage

    ```python

    model = load_from_hub(repo_id="{repo_id}", filename="q-learning.pkl")

    # Don't forget to check if you need to add additional attributes (is_slippery=False etc)
    env = gym.make(model["env_name"])
    ```
    """

    readme_path = repo_local_path / "README.md"
    readme = ""
    print(readme_path.exists())
    if readme_path.exists():
        with readme_path.open("r", encoding="utf8") as f:
            readme = f.read()
    else:
        readme = model_card

    with readme_path.open("w", encoding="utf-8") as f:
        f.write(readme)

    # Save our metrics to Readme metadata
    metadata_save(readme_path, metadata)

    # Step 6: Record a video
    video_path = repo_local_path / "replay.mp4"
    record_video(env, model["qtable"], video_path, video_fps)

    # Step 7. Push everything to the Hub
    api.upload_folder(
        repo_id=repo_id,
        folder_path=repo_local_path,
        path_in_repo=".",
    )

    print("Your model is pushed to the Hub. You can view your model here: ", repo_url)

def main(env_id):
    """
    Train and evaluate our deep RL model for lunar policy
    """
    virtual_display = Display(visible=0, size=(1400, 900))
    virtual_display.start()
    model_name = f"arkadyark-{env_id}"
    env = init_env(env_id)
    hparams = {
        "max_steps": 99,
        "n_training_episodes": 10000,
        "learning_rate": 0.7,
        "gamma": 0.95,
        "max_epsilon": 1.0,
        "min_epsilon": 0.05,
        "decay_rate": 0.0005,
    }
    eval_params = {
        "n_eval_episodes": 100,
        "eval_seed": [
            16,
            54,
            165,
            177,
            191,
            191,
            120,
            80,
            149,
            178,
            48,
            38,
            6,
            125,
            174,
            73,
            50,
            172,
            100,
            148,
            146,
            6,
            25,
            40,
            68,
            148,
            49,
            167,
            9,
            97,
            164,
            176,
            61,
            7,
            54,
            55,
            161,
            131,
            184,
            51,
            170,
            12,
            120,
            113,
            95,
            126,
            51,
            98,
            36,
            135,
            54,
            82,
            45,
            95,
            89,
            59,
            95,
            124,
            9,
            113,
            58,
            85,
            51,
            134,
            121,
            169,
            105,
            21,
            30,
            11,
            50,
            65,
            12,
            43,
            82,
            145,
            152,
            97,
            106,
            55,
            31,
            85,
            38,
            112,
            102,
            168,
            123,
            97,
            21,
            83,
            158,
            26,
            80,
            63,
            5,
            81,
            32,
            11,
            28,
            148,
        ]
    }
    qtable = train(env, **hparams)
    mean_reward, std_reward = eval_agent(qtable, env)
    print(f"Evaluation result: {mean_reward} ({std_reward})")


    model = {
        "env_id": env_id,
        "qtable": qtable,
        **hparams,
        **eval_params,
    }
    push_to_hub(f"arkadyark/q-{env_id}", model=model, env=env, video_fps=1, local_repo_path="hub")


if __name__ == "__main__":
    # main("FrozenLake-v1")
    main("Taxi-v3")
