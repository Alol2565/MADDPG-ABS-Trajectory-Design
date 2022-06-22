import numpy as np
import time, datetime
import matplotlib.pyplot as plt

class MetricLogger():
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_connected_users_plot = save_dir / "connected_users_plot.jpg"
        self.ep_bit_rate_plot = save_dir / "avg_bit_rate_plot.jpg"
        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()
        self.curr_connected_users = []
        self.curr_total_bit_rate = []
        self.ep_connected_users = []
        self.ep_total_bit_rate = []
        self.moving_avg_ep_connected_users = []
        self.moving_avg_ep_bit_rate = []


    def log_step(self, reward, connected_users, total_bit_rate):
        self.curr_ep_reward += reward
        self.curr_connected_users.append(connected_users)
        self.curr_total_bit_rate.append(total_bit_rate)

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_connected_users.append(np.mean(self.curr_connected_users))
        self.ep_total_bit_rate.append(np.mean(self.curr_total_bit_rate))
        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_connected_users = []
        self.curr_total_bit_rate = []

    # def record_nn(self, nn):
    #     with open(str(self.save_log) + str("_nn_specs"), "a") as f:
    #         f.write(str(summary(nn, (1,1,3))))


    def record_initials(self, memory_len, batch_size, exploration_rate_decay, burnin, learn_every, sync_every):
        with open(str(self.save_log) + str("_specs"), "a") as f:
            f.write(
                f"Memory len: {memory_len:15d}\nbatch size: {batch_size:15.3f}"
                f"\nexploration rate decay: {exploration_rate_decay:15.10f}\nburnin: {burnin:15.3f}\nlearn every: {learn_every:15.3f}"
                f"\nsync every: {sync_every:15.3f}\n"
            )

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-1:]), 3)
        mean_ep_connected_users = np.round(np.mean(self.ep_connected_users[-1:]), 3)
        mean_ep_total_bit_rate = np.round(np.mean(self.ep_total_bit_rate[-1:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_connected_users.append(mean_ep_connected_users)
        self.moving_avg_ep_bit_rate.append(mean_ep_total_bit_rate)
        self.record_time = time.time()

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Connected Users {mean_ep_connected_users} - "
            f"Mean Bit Rate {mean_ep_total_bit_rate} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_connected_users:15.3f}{mean_ep_total_bit_rate:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_connected_users", "ep_bit_rate"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.grid()
            plt.title(metric)
            plt.xlabel('Episodes')
            plt.ylabel(metric)
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()
