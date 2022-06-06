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
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
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


    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_loss_length = 0

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

    def record(self, episode, epsilon, step, mean_bit_rate):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-10:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-10:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-10:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.record_time = time.time()

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Bit Rate {mean_bit_rate:5.4f} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.grid()
            plt.title(metric)
            plt.xlabel('Episodes')
            plt.ylabel(metric)
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()
