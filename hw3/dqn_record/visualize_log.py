import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
    rewards = []
    steps = []
    average_len = 200
    with open(sys.argv[1], 'r') as f:
        for line in f:
            step, reward = line.strip('\n').split(',')
            rewards.append(float(reward))
            steps.append(int(step))

    avg_rewards = []
    for pos in range(len(rewards) - average_len):
        avg_rewards.append(sum(rewards[pos:pos+average_len]) / average_len)


    plt.plot(steps[average_len:], avg_rewards)
    plt.show()