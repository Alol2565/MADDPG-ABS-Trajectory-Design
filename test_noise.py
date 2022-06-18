from Agents.noise import OUActionNoise
import numpy as np


noise = OUActionNoise(mu=np.zeros(1), sigma=0.2, theta=0.15, dt=1e-2, x0=None)
noise.reset()

init_list = []
for i in range(10000):
    init_list.append(noise())
print(np.mean(init_list))


for i in range(int(1e6)):
    noise()

final_list = []
for i in range(10000):
    final_list.append(noise())
print(np.mean(final_list))
