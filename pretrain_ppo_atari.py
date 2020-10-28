import os
import numpy as np

from src.ppo_atari import PPO
from src.args import get_ppo_args
from src.utils import get_dirs, write_arguments, seed
from envs_utils.create_env import create_multiple_envs, create_single_env

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


if __name__ == '__main__':
    # set signle thread
    # os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['MKL_NUM_THREADS'] = '1'
    # get arguments
    args = get_ppo_args()
    model_dir, data_dir = get_dirs(args.dir_name)
    # video_dir = data_dir + '/video/train/'
    write_arguments(args, os.path.dirname(data_dir) + '/arguments.txt')
    # set seeds
    seed(args.seed)
    # start to create the environment
    envs = create_multiple_envs(args)

    # create trainer
    agent = PPO(envs.action_space.n, args)

    # start to train the network...
    evals = []
    best_eval_rew = -np.float("Inf")
    episode_rewards = np.zeros((args.num_workers, ), dtype=np.float32)
    final_rewards = np.zeros((args.num_workers, ), dtype=np.float32)
    # get the observation
    batch_ob_shape = (args.num_workers * args.nsteps, ) + envs.observation_space.shape
    obs = np.zeros((args.num_workers, ) + envs.observation_space.shape, dtype=envs.observation_space.dtype.name)
    obs[:] = envs.reset()
    dones = [False for _ in range(args.num_workers)]
    # get total number of updates
    num_updates = args.total_frames // (args.nsteps * args.num_workers)
    for update in range(num_updates):
        mb_obs, mb_rewards, mb_actions, mb_dones, mb_values = [], [], [], [], []
        if args.lr_decay:
            agent._adjust_learning_rate(args.lr, update, num_updates)
        for step in range(args.nsteps):
            # prepocessing the state for the task where there are some black features.
            obs += 5*np.ones((args.num_workers, ) + envs.observation_space.shape, dtype=envs.observation_space.dtype.name)
            # predict
            values, actions = agent.predict(obs, is_training=True)
            # start to store information
            mb_obs.append(np.copy(obs))
            mb_actions.append(actions)
            mb_dones.append(dones)
            mb_values.append(values)
            # start to excute the actions in the environment
            obs, rewards, dones, _ = envs.step(actions)
            mb_rewards.append(rewards)
            # clear the observation
            for n, done in enumerate(dones):
                if done:
                    obs[n] *= 0
            # process the rewards part -- display the rewards on the screen
            episode_rewards += rewards
            masks = np.array([0.0 if done_ else 1.0 for done_ in dones], dtype=np.float32)
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks
        
        # process the rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.float32)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        # compute the last state value
        last_values, _ = agent.predict(obs, is_training=True)
        # start to compute advantages...
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(args.nsteps)):
            if t == args.nsteps - 1:
                nextnonterminal = 1.0 - dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t + 1]
                nextvalues = mb_values[t + 1]
            delta = mb_rewards[t] + args.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + args.gamma * args.tau * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        # after compute the returns, let's process the rollouts
        mb_obs = mb_obs.swapaxes(0, 1).reshape(batch_ob_shape)
        mb_actions = mb_actions.swapaxes(0, 1).flatten()
        mb_returns = mb_returns.swapaxes(0, 1).flatten()
        mb_advs = mb_advs.swapaxes(0, 1).flatten()
        # start to update the network
        agent._update_network(mb_obs, mb_actions, mb_returns, mb_advs)

        # display the training information
        if update % args.display_interval == 0:
            mean_rewards = final_rewards.mean()
            print('Update: {} / {}, Rewards: {:.3f}, Min: {:.3f}, Max: {:.3f}'.format(
                update, num_updates, mean_rewards, final_rewards.min(), final_rewards.max()))
            # save data
            evals.append(mean_rewards)
            np.savez(data_dir+'/rewards.npz', evals)
            # save the model
            agent.save("final", model_dir)
            if mean_rewards > best_eval_rew:
                best_eval_rew = mean_rewards
                agent.save("best", model_dir)

    # close the environment
    envs.close()
