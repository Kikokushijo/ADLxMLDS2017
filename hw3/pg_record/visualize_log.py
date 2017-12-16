import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
    rewards = []
    average_len = 200
    with open(sys.argv[1], 'r') as f:
        for line in f:
            _, reward, _ = line.strip('\n').split(',')
            rewards.append(float(reward))

    avg_rewards = []
    for pos in range(len(rewards) - average_len):
        avg_rewards.append(sum(rewards[pos:pos+average_len]) / average_len)

    plt.plot(range(len(rewards) - average_len), avg_rewards)
    plt.title('Policy Gradient on Pong')
    plt.show()
