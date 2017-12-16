import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
    
    average_len = 2000
    labels = ['100', '500', '1000', '2000', '4000', '6000']
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    for i in range(1, 7):

        rewards = []
        steps = []

        with open(sys.argv[i], 'r') as f:
            for line in f:
                step, reward = line.strip('\n').split(',')
                rewards.append(float(reward))
                steps.append(int(step))

        avg_rewards = []
        for pos in range(len(rewards) - average_len):
            avg_rewards.append(sum(rewards[pos:pos+average_len]) / average_len)

        plt.plot(steps[average_len:], avg_rewards, color=colors[i-1], label=labels[i-1])
    plt.legend()
    plt.title('DQN on Breakout\nwith different target network update frequency')
    plt.xlabel('number of time steps')
    plt.ylabel('average reward in last %d episodes' % (average_len))
    plt.show()
