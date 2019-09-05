import numpy as np 
from PIL import Image 
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
# from google.colab.patches import cv2_imshow

style.use("ggplot")

SIZE = 10
EPISODES  = 25000
epsilon = 0.9
EPS_DECAY = 0.9998
DISCOUNT = 0.9
LEARNING_RATE = 0.1

# Metrics

SHOW_EVERY = 3000

# reward systems

ENEMY_PENALTY = 300
FOOD_REWARD = 25
MOVE_PENALTY = 1

# provision for an existing_q_table

start_q_table = None

# Colour of objects for our environment

PLAYER = 1
FOOD = 2
ENEMY = 3

colour_dict = {1: (255,240,0),
                2: (0, 255, 0),
                3: (0, 0, 255)}

# create object classes for the environment.

class block:

    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def move(self, x = False, y = False):
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x
        
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x > SIZE -1:
            self.x = SIZE -1

        if self.y < 0:
            self.y = 0
        elif self.y > SIZE -1:
            self.y = SIZE -1

    def action(self, choice):
        if choice == 0:
            self.move(x = -1, y = -1)
        elif choice == 1:
            self.move(x = 1, y = -1)
        elif choice == 2:
            self.move(x = 1, y = 1)
        elif choice == 3:
            self.move(x = -1, y = 1)

# initiating q table

if start_q_table is None:
    q_table = {}
    for x1 in range(-SIZE + 1, SIZE):
        for y1 in range(-SIZE + 1, SIZE):
            for x2 in range(-SIZE + 1, SIZE):
                for y2 in range(-SIZE + 1, SIZE):
                    q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]

else:
    with open(start_q_table, 'rb') as f:
        q_table = pickle.load(f)

# start training on episodes

episode_rewards = []
goal_steps = 200

for episode in range(EPISODES):
    player = block()
    food  = block()
    enemy = block()
    episode_reward = 0

    if episode % SHOW_EVERY == 0:
        print(f"on # {episode}, the epsilon = {epsilon}")
        show = True
    else:
        show = False

    for step in range(goal_steps):
        position = (player - food, player - enemy)

        if np.random.random() > epsilon:
            action = np.argmax(q_table[position])
        else:
            action = np.random.randint(0, 4)
        
        player.action(action)

        # give rewards
        if player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        elif player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        else:
            reward = -MOVE_PENALTY

        # episode_reward += reward


        new_position = (player - food, player - enemy)

        current_q = q_table[position][action]
        max_future_q = np.max(q_table[new_position])

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        q_table[position][action] = new_q

        episode_reward += reward

        # show the env visulally

        if show:

            env = np.zeros((SIZE, SIZE, 3), dtype = np.uint8)
            env[player.y][player.x] = colour_dict[PLAYER]
            env[food.y][food.x] = colour_dict[FOOD]
            env[enemy.y][enemy.x] = colour_dict[ENEMY]

            img = Image.fromarray(env, 'RGB')
            img.resize((300, 300))
            cv2.imshow(np.array(img))

            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break

            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

# save the q table as a pickle file 
with open(f"qtable--{int(time.time())}.pickle", 'wb') as f:
  pickle.dump(q_table, f)



