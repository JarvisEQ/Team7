
import tensorflow as tf

from anyrl.algos import DQN
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.models import rainbow_models
from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer
import gym_remote.exceptions as gre

from sonic_util import AllowBacktracking, make_env

def main():
    server_address = os.getenv('SERVER_ADDRESS', 'server')
    pub_socket = int(os.getenv('PUB_SOCKET', pub_socket))
    if pub_socket is None:
        raise Exception('Pub socket not set')

    # print(f'Pub socket is {pub_socket}')

    env_config = {
        'await_connection_time': 120,
        'server_address':  server_address,
        'pub_socket': pub_socket,
        'sub_socket': '5563',
    }

    _env_name = os.getenv('ENV_NAME', 'everglades')

    render_image = os.getenv('RENDER_IMAGE', 'false').lower() == 'true'
    viewer = None

    env_name = ENV_MAP[_env_name.lower()]
    env = gym.make(env_name, env_config=env_config)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    with tf.Session(config=config) as sess:
        dqn = DQN(*rainbow_models(sess,
                                  env.action_space.n,
                                  gym_space_vectorizer(env.observation_space),
                                  min_val=-200,
                                  max_val=200))
        player = NStepPlayer(BatchedPlayer(env, dqn.online_net), 3)
        optimize = dqn.optimize(learning_rate=1e-4)
        sess.run(tf.global_variables_initializer())
        dqn.train(num_steps=2000000, # Make sure an exception arrives before we stop.
                  player=player,
                  replay_buffer=PrioritizedReplayBuffer(500000, 0.5, 0.4, epsilon=0.1),
                  optimize_op=optimize,
                  train_interval=1,
                  target_interval=8192,
                  batch_size=32,
                  min_buffer_size=20000)
