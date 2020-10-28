import argparse


def get_td3_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="duckietown", type=str)
    parser.add_argument("--seed", default=123, type=int, help="Sets Gym, PyTorch and Numpy seeds")
    parser.add_argument("--algo", default="td3", type=str)
    parser.add_argument("--start_timesteps", default=10000, type=int, help="How many time steps purely random policy is run for")
    parser.add_argument("--eval_freq", default=5000, type=int, help="How often (time steps) we evaluate")
    parser.add_argument("--max_timesteps", default=500000, type=int, help="Max time steps to run environment for")
    parser.add_argument("--expl_noise", default=0.1, type=float, help="Std of Gaussian exploration noise")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for both actor and critic")
    parser.add_argument("--discount", default=0.99, type=float, help="Discount factor")
    parser.add_argument("--tau", default=0.005, type=float, help="Target network update rate")
    parser.add_argument("--policy_noise", default=0.2, type=float, help="Noise added to target policy during critic update")
    parser.add_argument("--noise_clip", default=0.5, type=float, help="Range to clip target policy noise")
    parser.add_argument("--policy_freq", default=2, type=int, help="Frequency of delayed policy updates")
    parser.add_argument("--env_timesteps", default=500, type=int)
    parser.add_argument("--replay_buffer_max_size", default=10000, type=int, help="Maximum number of steps to keep in the replay buffer")
    parser.add_argument("--critic_lr", default=1e-3, type=float)
    parser.add_argument("--actor_lr", default=1e-4, type=float)
    parser.add_argument("--save_models", default=True, action="store_true", help="Whether or not models are saved")
    parser.add_argument("--model_dir", default=None, type=str)
    
    parser.add_argument("--dir_name", default="td3", type=str)
    parser.add_argument("--map_name", default="loop_empty", type=str)
    parser.add_argument("--actor_net_type", default="cnn", type=str, help="select from [cnn, dense, unet, ResnetLW, MobilenetLW, Deeplabv3, fcdensenet]")
    parser.add_argument("--net_layers", default=4, type=int, help="select from [3, 4, 5]")
    parser.add_argument("--critic_net_type", default="cnn", type=str, help="select from [cnn, dense, unet]")
    parser.add_argument("--fc_hid_size", default=256, type=int)
    parser.add_argument("--spec_init", default=False, action = "store_true")
    parser.add_argument("--noisy_lin", default=False, action = "store_true")
    parser.add_argument("--noisy_std", default=0.5, type=float)
    parser.add_argument("--dropoutlin", default=False, action = "store_true")
    parser.add_argument("--dropoutconv", default=False, action = "store_true")
    parser.add_argument("--dropout_scale", default=0.3, type=float)
    parser.add_argument('--num_channel', type=int, default=3, help='the number of image channel')
    
    # For training the mask net
    parser.add_argument("--reg_scale", default=0.04, type=float)
    parser.add_argument("--mask_lr", default=0.01, type=float)
    parser.add_argument("--dataset_size", default=50000, type=int)
    
    return parser.parse_args()


def get_sac_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="duckietown", type=str)
    parser.add_argument("--seed", default=123, type=int, help="Sets Gym, PyTorch and Numpy seeds")
    parser.add_argument("--algo", default="sac", type=str)
    parser.add_argument("--start_timesteps", default=10000, type=int, help="How many time steps purely random policy is run for")
    parser.add_argument("--eval_freq", default=5000, type=int, help="How often (time steps) we evaluate")
    parser.add_argument("--max_timesteps", default=500000, type=int, help="Max time steps to run environment for")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size for both actor and critic")
    parser.add_argument("--discount", default=0.99, type=float, help="Discount factor")
    parser.add_argument("--tau", default=0.005, type=float, help="Target network update rate")
    parser.add_argument("--env_timesteps", default=500, type=int)
    parser.add_argument("--replay_buffer_max_size", default=10000, type=int, =help="Maximum number of steps to keep in the replay buffer")
    parser.add_argument("--critic_lr", default=3e-4, type=float)
    parser.add_argument("--actor_lr", default=3e-4, type=float)
    parser.add_argument('--alpha', default=0.2, type=float, help="Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)")
    parser.add_argument("--save_models", default=True, action="store_true", help="Whether or not models are saved")
    parser.add_argument("--model_dir", default=None, type=str)
    
    parser.add_argument("--dir_name", default="sac", type=str)
    parser.add_argument("--map_name", default="loop_empty", type=str)
    parser.add_argument("--actor_net_type", default="cnn", type=str, help="select from [cnn, dense, unet, ResnetLW, MobilenetLW, Deeplabv3, fcdensenet]")
    parser.add_argument("--net_layers", default=4, type=int, help="select from [3, 4, 5]")
    parser.add_argument("--critic_net_type", default="cnn", type=str, help="select from [cnn, dense, unet]")
    parser.add_argument("--fc_hid_size", default=256, type=int)
    parser.add_argument("--spec_init", default=False, action="store_true")
    parser.add_argument("--noisy_lin", default=False, action="store_true")
    parser.add_argument("--noisy_std", default=0.5, type=float)
    parser.add_argument("--dropoutlin", default=False, action="store_true")
    parser.add_argument("--dropoutconv", default=False, action="store_true")
    parser.add_argument("--dropout_scale", default=0.3, type=float)
    parser.add_argument('--num_channel', type=int, default=3, help='the number of image channel')
    
    # For training the mask net
    parser.add_argument("--reg_scale", default=0.04, type=float)
    parser.add_argument("--mask_lr", default=0.01, type=float)
    parser.add_argument("--dataset_size", default=50000, type=int)
    
    return parser.parse_args()


def get_ppo_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="duckietown", type=str, help='duckietown or atari')
    parser.add_argument("--map_name", default="loop_empty", type=str, help='the map name for duckietown')
    parser.add_argument('--task_name', type=str, default='AssaultNoFrameskip-v4', help='the task name for Atari')
    parser.add_argument("--algo", default="ppo", type=str)
    parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor of RL')
    parser.add_argument('--seed', type=int, default=123, help='the random seeds')
    parser.add_argument('--num_workers', type=int, default=8, help='the number of workers to collect samples')
    parser.add_argument('--batch_size', type=int, default=4, help='the batch size of updating')
    parser.add_argument('--lr', type=float, default=2.5e-4, help='learning rate of the algorithm')
    parser.add_argument('--epoch', type=int, default=4, help='the epoch during training')
    parser.add_argument('--nsteps', type=int, default=128, help='the steps to collect samples')
    parser.add_argument("--env_timesteps", default=500, type=int)
    parser.add_argument('--vloss_coef', type=float, default=0.5, help='the coefficient of value loss')
    parser.add_argument('--ent_coef', type=float, default=0.01, help='the entropy loss coefficient')
    parser.add_argument('--tau', type=float, default=0.95, help='gae coefficient')
    parser.add_argument('--total_frames', type=int, default=20000000, help='the total frames for training')
    parser.add_argument('--eps', type=float, default=1e-5, help='param for adam optimizer')
    parser.add_argument('--clip', type=float, default=0.1, help='the ratio clip param')
    parser.add_argument('--lr_decay', action='store_true', help='if using the learning rate decay during decay')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='grad norm')
    parser.add_argument('--display_interval', type=int, default=5, help='the interval that display log information')
    parser.add_argument('--eval_interval', type=int, default=40, help='the interval that eval the policy')
    parser.add_argument("--save_models", default=True, action="store_true", help="Whether or not models are saved")
    parser.add_argument("--model_dir", default=None, type=str)
    
    parser.add_argument("--dir_name", default="ppo", type=str, help='the folder to save model and data')
    parser.add_argument("--actor_net_type", default="cnn", type=str, help="select from [cnn, dense, unet, ResnetLW, MobilenetLW, Deeplabv3, fcdensenet]")
    parser.add_argument("--net_layers", default=4, type=int, help="select from [3, 4, 5]")
    parser.add_argument("--critic_net_type", default="cnn", type=str, help="select from [cnn, dense, unet]")
    parser.add_argument("--fc_hid_size", default=256, type=int)
    parser.add_argument("--spec_init", default=False, action="store_true")
    parser.add_argument("--noisy_lin", default=False, action="store_true")
    parser.add_argument("--noisy_std", default=0.5, type=float)
    parser.add_argument("--dropoutlin", default=False, action="store_true")
    parser.add_argument("--dropoutconv", default=False, action="store_true")
    parser.add_argument("--dropout_scale", default=0.3, type=float)
    parser.add_argument("--is_discrete", default=False, action="store_true")
    parser.add_argument('--num_channel', type=int, default=3, help="the number of image channel")
    
    # For training the mask net
    parser.add_argument("--reg_scale", default=0.04, type=float)
    parser.add_argument("--mask_lr", default=0.01, type=float)
    parser.add_argument("--dataset_size", default=50000, type=int)
    
    return parser.parse_args()
