import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


## Create Lever Class
class Lever:
    def __init__(self):

        # A lever has a distribution which it gives rewards
        self.mu = np.random.uniform(-5, 5, 1)
        self.sd = np.random.uniform(0.1, 3, 1)

        self.Qa = 0  # Estimate value
        self.Na = 0  # Number of actions(pulls)
        self.Ha = 0  # Preference H_t(a)

        self.rewards = []

    def act(self, alpha=None):
        reward = np.random.normal(self.mu, self.sd)
        self.rewards.append(reward)
        self.Na += 1

        if alpha is None:
            self.Qa = self.Qa + (1/self.Na) * (reward - self.Qa)
        else:
            self.Qa = self.Qa + alpha * (reward - self.Qa)

        return reward

def plotlevers(levers):
    kwargs = dict(kde=True, stat="density", common_norm=False, alpha=0.6, linewidth=2)
    
    colors = ["dodgerblue", "orange", "deeppink", "dimgrey", "lightcoral",
              "goldenrod", "limegreen", "lightseagreen", "magenta", "chocolate"]
    labels = ["Bandit1", "Bandit2", "Bandit3", "Bandit4", "Bandit5",
              "Bandit6", "Bandit7", "Bandit8", "Bandit9", "Bandit10"]
    
    plt.figure(figsize=(10,7), dpi=80)
    i = 0
    for l in levers:
        convert_rewards = [arr.item() for arr in l.rewards]
        sns.histplot(convert_rewards, color=colors[i], label=labels[i], **kwargs)
        i = i+1
    plt.xlim(-15, 15)
    plt.legend()
    plt.show()
    plt.clf()
    

def sortingFunction(l):
    return l.Qa


# Write code to run the simulation
def run_greedy(nlevers=4, epsilon=0.1, iterations=2000):
    levers = []
    for i in range(nlevers):
        l = Lever()
        levers.append(l)

    for i in range(iterations):
        if (np.random.rand() <= epsilon):
            l = np.random.choice(levers)  # Explore
        else:
            templevers = levers.copy()
            np.random.shuffle(templevers)  # Shuffling levers to break ties
            templevers.sort(reverse=True, key=sortingFunction)
            l = templevers[0]

        l.act()

    total_rewards = sum([sum(l.rewards) for l in levers])

    # plotlevers(levers)
    # for i in levers:
    #     print("mu=", i.mu, ", Qa=", i.Qa, ", Na=", i.Na, sep=" ")

    return total_rewards


def run_optimistic(nlevers=4, initial_qa=5, epsilon=0.1, iterations=2000):
    levers = []
    for i in range(nlevers):
        l = Lever()
        l.Qa = initial_qa  # High starting value
        levers.append(l)

    for i in range(iterations):
        if (np.random.rand() <= epsilon):
            l = np.random.choice(levers)  # Explore
        else:
            templevers = levers.copy()
            np.random.shuffle(templevers)  # Shuffling levers to break ties
            templevers.sort(reverse=True, key=sortingFunction)
            l = templevers[0]

        l.act(alpha=0.1)

    total_rewards = sum([sum(l.rewards) for l in levers])

    # plotlevers(levers)
    # for i in levers:
    #     print("mu=", i.mu, ", Qa=", i.Qa, ", Na=", i.Na, sep=" ")

    return total_rewards


def run_ucb(nlevers=4, c=1, iterations=2000):
    levers = []
    for i in range(nlevers):
        l = Lever()
        levers.append(l)

    for t in range(1, iterations + 1):  # ln(0) is not defined
        ucb_values = []
        for l in levers:
            if l.Na == 0:
                ucb_values.append(float('inf'))
            else:
                ucb = l.Qa.item() + c * np.sqrt(np.log(t) / l.Na)
                ucb_values.append(ucb)

        # ucb_values.sort(reverse=True, key=sortingFunction)
        l = levers[np.argmax(ucb_values)]  # Return the index of the lever with highest Qa
        l.act(alpha=None)

    total_rewards = sum([sum(l.rewards) for l in levers])

    # plotlevers(levers)
    # for i in levers:
    #     print("mu=", i.mu, ", Qa=", i.Qa, ", Na=", i.Na, sep=" ")

    return total_rewards


def run_gradient_bandit(nlevers=4, alpha=0.1, iterations=2000):
    levers = []
    for i in range(nlevers):
        l = Lever()
        levers.append(l)

    avg_reward = 0
    for t in range(1, iterations + 1):
        # Applied softmax function for each lever
        # e^H(a1)/sum(e^H(a1) + e^H(a2) + e^H(a3) + ...)
        sum_exp_Ha = sum(np.exp(l.Ha) for l in levers)
        softmax_probs = [np.exp(l.Ha) / sum_exp_Ha for l in levers]
        softmax_probs = np.array(softmax_probs).flatten()

        selected_lever_index = np.random.choice(nlevers, p=softmax_probs)
        selected_lever = levers[selected_lever_index]

        # Get reward
        reward = selected_lever.act()
        # Update average reward
        avg_reward += (reward - avg_reward) / t

        # Update preference Ha value
        for i, l in enumerate(levers):
            if i == selected_lever_index:
                l.Ha += alpha * (reward - avg_reward) * (1 - softmax_probs[i])
            else:
                l.Ha -= alpha * (reward - avg_reward) * softmax_probs[i]

    # for i in levers:
    #     print("mu=", i.mu, ", Ha=", i.Ha, ", Na=", i.Na, sep=" ")

    total_rewards = sum([sum(l.rewards) for l in levers])
    return total_rewards


if __name__ == '__main__':
    times = 2000
    print("Running results...")

    print("\nGreedy Method...")
    sum_rewards = 0
    for i in range(times):
        sum_rewards += run_greedy()
    print(f"Average reward over 2000 runs: {sum_rewards/times}.")
    # --------------------------------------------------------------
    print("\nOptimistic Method...")
    sum_rewards = 0
    for i in range(times):
        sum_rewards += run_optimistic()
    print(f"Average reward over 2000 runs: {sum_rewards/times}.")
    # --------------------------------------------------------------
    print("\nUpper-Confidence-Bound Method...")
    sum_rewards = 0
    for i in range(times):
        sum_rewards += run_ucb()
    print(f"Average reward over 2000 runs: {sum_rewards/times}.")
    # --------------------------------------------------------------
    print("\nGradient Bandit Method...")
    sum_rewards = 0
    for i in range(times):
        sum_rewards += run_gradient_bandit()
    print(f"Average reward over 2000 runs: {sum_rewards/times}.")
