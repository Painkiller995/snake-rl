"""
script for training the agent for snake using various methods
"""

# run on cpu

import time

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from agent import DeepQLearningAgent
from game_environment import SnakeNumpy
from model_config.model_17_1 import config
from utils import play_game2

# some global variables
tf.random.set_seed(42)

# define no of episodes, logging frequency
episodes = 2 * (10**5)
log_frequency = 500
games_eval = 8

# setup the agent
agent = DeepQLearningAgent(
    board_size=config["board_size"],
    frames=config["frames"],
    n_actions=config["n_actions"],
    buffer_size=config["buffer_size"],
    version=config["version"],
)

# check in the same order as class hierarchy
if isinstance(agent, DeepQLearningAgent):
    print("Agent is a DeepQLearningAgent")

# setup the epsilon range and decay rate for epsilon
# define rewrad type and update frequency, see utils for more details
epsilon, epsilon_end = 1, 0.01
reward_type = "current"
sample_actions = False
n_games_training = 8 * 16
decay = 0.97
if config["supervised"]:
    # lower the epsilon since some starting policy has already been trained
    epsilon = 0.01
    # load the existing model from a supervised method
    # or some other pretrained model
    agent.load_model(file_path="models/{:s}".format(config["version"]))
    # agent.set_weights_trainable()


# decay = np.exp(np.log((epsilon_end/epsilon))/episodes)

# use only for DeepQLearningAgent

# play some games initially to fill the buffer
# or load from an existing buffer (supervised)
if config["supervised"]:
    try:
        agent.load_buffer(file_path="models/{:s}".format(config["version"]), iteration=1)
    except FileNotFoundError:
        pass
else:
    # setup the environment
    games = 512
    env = SnakeNumpy(
        board_size=config["board_size"],
        frames=config["frames"],
        max_time_limit=config["max_time_limit"],
        games=games,
        frame_mode=True,
        obstacles=config["obstacles"],
        version=config["version"],
    )
    ct = time.time()
    _ = play_game2(
        env,
        agent,
        n_actions=config["n_actions"],
        n_games=games,
        record=True,
        epsilon=epsilon,
        verbose=True,
        reset_seed=False,
        frame_mode=True,
        total_frames=games * 64,
    )
    print("Playing {:d} frames took {:.2f}s".format(games * 64, time.time() - ct))

env = SnakeNumpy(
    board_size=config["board_size"],
    frames=config["frames"],
    max_time_limit=config["max_time_limit"],
    games=n_games_training,
    frame_mode=True,
    obstacles=config["obstacles"],
    version=config["version"],
)
env2 = SnakeNumpy(
    board_size=config["board_size"],
    frames=config["frames"],
    max_time_limit=config["max_time_limit"],
    games=games_eval,
    frame_mode=True,
    obstacles=config["obstacles"],
    version=config["version"],
)

# training loop
model_logs = {"iteration": [], "reward_mean": [], "length_mean": [], "games": [], "loss": []}
for index in tqdm(range(episodes)):
    # make small changes to the buffer and slowly train
    _, _, _ = play_game2(
        env,
        agent,
        n_actions=config["n_actions"],
        epsilon=epsilon,
        n_games=n_games_training,
        record=True,
        sample_actions=sample_actions,
        reward_type=reward_type,
        frame_mode=True,
        total_frames=n_games_training,
        stateful=True,
    )
    loss = agent.train_agent(batch_size=64, num_games=n_games_training, reward_clip=True)

    # check performance every once in a while
    if (index + 1) % log_frequency == 0:
        # keep track of agent rewards_history
        current_rewards, current_lengths, current_games = play_game2(
            env2,
            agent,
            n_actions=config["n_actions"],
            n_games=games_eval,
            epsilon=-1,
            record=False,
            sample_actions=False,
            frame_mode=True,
            total_frames=-1,
            total_games=games_eval,
        )

        model_logs["iteration"].append(index + 1)
        model_logs["reward_mean"].append(round(int(current_rewards) / current_games, 2))
        # model_logs['reward_dev'].append(round(np.std(current_rewards), 2))
        model_logs["length_mean"].append(round(int(current_lengths) / current_games, 2))
        model_logs["games"].append(current_games)
        model_logs["loss"].append(loss)
        pd.DataFrame(model_logs)[["iteration", "reward_mean", "length_mean", "games", "loss"]].to_csv(
            "model_logs/{:s}.csv".format(config["version"]), index=False
        )

    # copy weights to target network and save models
    if (index + 1) % log_frequency == 0:
        agent.update_target_net()
        agent.save_model(file_path="models/{:s}".format(config["version"]), iteration=(index + 1))
        # keep some epsilon alive for training
        epsilon = max(epsilon * decay, epsilon_end)
