import numpy as np
from tqdm import tqdm
from environment.env import Environnement
from matplotlib import pyplot as plt


class QLearningAgent():
    """Class for Q-Learning & Double Q-Learning Algorithm"""

    def __init__(self, environment: Environnement, gamma:float = 0.99, learning_rate: float = 1e-2, initial_q_value: float = -1):
        """
        Class for Q-Learning Algorithm

        """
        
        self.environment = environment
        self.n_states = environment.state_space.n_states
        self.n_actions = environment.action_space.n_actions
        
        self.gamma = gamma
        self.learning_rate = learning_rate
        
        self.Q_matrix = initial_q_value * np.ones((self.n_states, self.n_actions))
        self.initial_value_Q_matrix = initial_q_value

        self.policy = np.zeros((self.n_states, self.n_actions))
        
        
    def update_Q_matrix(self, reward: float, current_state: int, next_state: int, action: int) -> None:
        """Updates the Q-matrix using the Bellman equation.

        Args:
            reward (float): _description_
            current_state (int): _description_
            next_state (int): _description_
            action (int): _description_
        Returns:
            None
        """
        # The old Q value
        old_Q_value = self.Q_matrix[current_state, action]
        # The new Q value (learned value)
        new_Q_value = old_Q_value + self.learning_rate * (reward + self.gamma * np.max(self.Q_matrix[next_state]) - old_Q_value)
        # Update the Q-matrix of the current state and action pair
        self.Q_matrix[current_state, action] = new_Q_value
        
    def choose_action(self, state_index, epsilon):
        """
        Choose an action following Epsilon Greedy Policy.
        """
        if np.random.random() < epsilon:
            # Explore: choose a random action
            action_index = self.environment.action_space.sample()
        else:
            # Exploit: choose the action with the highest Q-value for the current state
            action_index = np.argmax(self.Q_matrix[state_index])
        return action_index
    
    def train(self, n_episodes: int = 2000, n_time_steps: int = 5000, epsilon_decay: float = 0.999, epsilon_min: float = 0.01):
        """
        Train the Q-Learning agent.
        
        Parameters
        ----------
            - n_episodes : Number of Episodes
            - n_time_steps : Maximum number of time steps per episode
            - epsilon_decay : Decay Rate for the Exploration Rate
            - epsilon_min : Minimum Exploration Rate
        """

        print("[INFO] Q-Learning Training: Process Initiated ... ")
        print(f'[INFO] The state space is of size {self.n_states * self.n_actions}.')
        
        avg_rewards = []
        
        epsilon = 1

        for episode in tqdm(range(n_episodes)):
            
            # Generate a Random Initial State
            _ = self.environment.reset()

            percentage_unvisited_states = self.computes_percentage_unvisited_states()

            reward_episode = []
            
            for time_step in range(n_time_steps):
                # Get the State Index
                current_state_index = self.environment.state_space.get_state_index()
                # Choose an action following Epsilon Greedy Policy
                action_index = self.choose_action(current_state_index, epsilon)
                # Update State
                next_state_index, reward = self.environment.step(action_index)
                # Update Q(s, a)
                self.update_Q_matrix(reward, current_state_index, next_state_index, action_index)
                reward_episode.append(reward)

            avg_reward = np.mean(reward_episode)
            avg_rewards.append(avg_reward)
            
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            
            print(f'[INFO] Episode: {episode+1}/{n_episodes}, Average Reward: {avg_reward}, Epsilon: {epsilon:.2f}, Percentage of unvisited states: {percentage_unvisited_states:.2f}')
            print('---------------------------------------------------')

        print("[INFO] Q-Learning Training : Process Completed !")
        
        # Extract policy
        for s in range(self.n_states):
            best_action_index = np.argmax(self.Q_matrix[s])
            self.policy[s, best_action_index] = 1.0
        print(f"[INFO] Policy : {self.policy}")

        plt.plot(avg_rewards)
        plt.title("Convergence of Q-Learning Algorithm")
        plt.xlabel("Iteration")
        plt.ylabel("Average Reward")
        plt.savefig("./figures/convergence_q_learning.png")
    
    def computes_percentage_unvisited_states(self) -> float:
        """Computes the percentage of unvisited states in the Q-matrix.

        Returns:
            percentage_unvisited_states (float): The percentage of unvisited states in the Q-matrix.
        """
        # compute the number of unvisited states
        number_of_unvisited_states = np.count_nonzero(self.Q_matrix == self.initial_value_Q_matrix)
        # compute the percentage of unvisited states
        percentage_unvisited_states = 100 * number_of_unvisited_states / self.Q_matrix.size
        return percentage_unvisited_states
