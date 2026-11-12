# script to visualize how agent plays a game
# useful to study different iterations


from agent import DeepQLearningAgent
from game_environment import Snake
from model_config.model_17_1 import config
from utils import visualize_game

version = config["version"]

iteration_list = [163500]
max_time_limit = 398

# setup the environment
env = Snake(
    board_size=config["board_size"],
    frames=config["frames"],
    max_time_limit=config["max_time_limit"],
    obstacles=config["obstacles"],
    version=config["version"],
)
s = env.reset()
n_actions = env.get_num_actions()

# setup the agent
agent = DeepQLearningAgent(
    board_size=config["board_size"],
    frames=config["frames"],
    n_actions=n_actions,
    buffer_size=10,
    version=config["version"],
)

for iteration in iteration_list:
    agent.load_model(file_path="models/{:s}".format(version), iteration=iteration)

    for i in range(5):
        visualize_game(
            env,
            agent,
            path="images/game_visual_{:s}_{:d}_14_ob_{:d}.mp4".format(version, iteration, i),
            debug=False,
            animate=True,
            fps=12,
        )
