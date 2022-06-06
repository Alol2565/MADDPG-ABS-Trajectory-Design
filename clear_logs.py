import os

dir_name = "log/env"
test = os.listdir(dir_name)

for item in test:
    if item.endswith(".log"):
        os.remove(os.path.join(dir_name, item))

dir_name = "log/rl"
test = os.listdir(dir_name)

for item in test:
    if item.endswith(".log"):
        os.remove(os.path.join(dir_name, item))

dir_name = "render_results"
test = os.listdir(dir_name)

for item in test:
    if item.endswith(".png"):
        os.remove(os.path.join(dir_name, item))