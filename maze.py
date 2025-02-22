import random

# Define wall combinations based on your rules
WALL_COMBINATIONS = {
    0: "no_wall",
    1: "up_wall",
    2: "down_wall",
    3: "up_down_wall",
    4: "left_wall",
    5: "up_left_wall",
    6: "down_left_wall",
    7: "up_down_left_wall",
    8: "right_wall",
    9: "up_right_wall",
    10: "down_right_wall",
    11: "up_down_right_wall",
    12: "left_right_wall",
    13: "up_left_right_wall",
    14: "down_left_right_wall",
    15: "up_down_left_right_wall"
}

# Define possible actions: up, down, left, right
ACTIONS = {
    0: (0, -1),  # Left
    1: (0, 1),   # Right
    2: (-1, 0),  # Up
    3: (1, 0)    # Down
}

WALLCHECKS = {
        0: 4,  # Left wall
        1: 8,  # Right wall
        2: 1,  # Up wall
        3: 2   # Down wall
    }

def break_wall(grid, x, y, nx, ny):
    """Break walls between two cells."""
    if nx == x - 1:  # Moving up
        grid[x][y] -= 1
        grid[nx][ny] -= 2
    elif nx == x + 1:  # Moving down
        grid[x][y] -= 2
        grid[nx][ny] -= 1
    elif ny == y - 1:  # Moving left
        grid[x][y] -= 4
        grid[nx][ny] -= 8
    elif ny == y + 1:  # Moving right
        grid[x][y] -= 8
        grid[nx][ny] -= 4

def recursive_backtracking(grid, x, y):
    """Recursive backtracking algorithm to generate a maze."""
    directions = [(0, -1), (-1, 0), (0, 1), (1, 0)]  # Left, Up, Right, Down
    random.shuffle(directions)  # Randomize directions to ensure variety

    for dx, dy in directions:
        nx, ny = x + dx, y + dy

        # Check if the neighbor is within bounds and unvisited (all walls intact)
        if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == 15:
            break_wall(grid, x, y, nx, ny)   # Break the wall between current cell and neighbor
            recursive_backtracking(grid, nx, ny)   # Recur on the neighbor

def generate_maze(rows, cols):
    """Generate a maze using recursive backtracking."""
    # Initialize the grid with all walls (15 means all walls are present)
    grid = [[15 for _ in range(cols)] for _ in range(rows)]

    # Start maze generation from a random cell
    start_x, start_y = random.randint(0, rows - 1), random.randint(0, cols - 1)
    recursive_backtracking(grid, start_x, start_y)

    # Select origin and target points
    origin_coords = (start_x, start_y)
    
    # Ensure the target is far from the origin for better paths
    while True:
        target_x = random.randint(0, rows - 1)
        target_y = random.randint(0, cols - 1)
        if abs(target_x - start_x) + abs(target_y - start_y) > max(rows // 2, cols // 2):
            target_coords = (target_x, target_y)
            break

    return grid, origin_coords, target_coords

def convert_to_alpha_format(grid, origin_coords, target_coords):
    """Convert the maze grid into the dumb format."""
    rows, cols = len(grid), len(grid[0])
    
    def cell_to_walls(cell_value):
        """Convert a cell's integer value to its wall type."""
        return WALL_COMBINATIONS[cell_value]

    alpha_format = ""
    
    for r in range(rows):
        for c in range(cols):
            coord = f"<|{r}-{c}|>"
            
            wall_type = cell_to_walls(grid[r][c])
            
            # Add origin or target markers **AFTER** the wall type
            if (r, c) == origin_coords:
                marker = "<|origin|>"
            elif (r, c) == target_coords:
                marker = "<|target|>"
            else:
                marker = "<|blank|>"
            
            alpha_format += f"{coord}<|{wall_type}|>{marker}"
        
        alpha_format += "\n"
    
    return alpha_format

import numpy as np
import matplotlib.pyplot as plt
import tqdm
import imageio

class MazeEnvironment:
    def __init__(self, maze_grid, origin_coords, target_coords):
        self.maze_grid = maze_grid
        self.origin_coords = origin_coords
        self.target_coords = target_coords
        self.current_state = origin_coords

    def reset(self):
        """Reset the environment to the initial state."""
        self.current_state = self.origin_coords
        return self.current_state

    def step(self, action):
        """Take a step in the environment based on the given action."""
        dx, dy = ACTIONS[action]
        nx, ny = self.current_state[0] + dx, self.current_state[1] + dy

        # Check if the new state is within bounds
        if 0 <= nx < len(self.maze_grid) and 0 <= ny < len(self.maze_grid[0]):
            cell_value = self.maze_grid[self.current_state[0]][self.current_state[1]]
            
            """
            This 'cell_value & WALLCHECKS[action]' check works by getting the int value of the wall
            and using bitwise AND to check if there is a wall in the direction of the action.
            Logic:
            0001 - Up wall
            0010 - Down wall
            0100 - Left wall
            1000 - Right wall
            So if we want to check if there is a wall to the left, we do cell_value & 0100
            If the result is not 0, then there is a wall to the left.
            This is a bitwise operation, so it is very fast.
            """
            if cell_value & WALLCHECKS[action]: # Check if there is a wall in the direction of the action
                return self.current_state, -1, False, {}  # Hit a wall
            else:
                self.current_state = (nx, ny)
                if self.current_state == self.target_coords:
                    return self.current_state, 10, True, {}  # Reached target
                else:
                    return self.current_state, -1, False, {}  # Valid move
        else:
            return self.current_state, -1, False, {}  # Out of bounds

    def render(self, ax=None):
        """Render the current state of the environment."""
        if ax is None:
            fig, ax = plt.subplots()
        rows, cols = len(self.maze_grid), len(self.maze_grid[0])
        
        # Clear the axis to avoid overlapping legends
        ax.clear()
        
        # Draw the grid
        for r in range(rows):
            for c in range(cols):
                cell_value = self.maze_grid[r][c]
                
                # Draw walls
                if cell_value & 1:  # Up wall
                    ax.plot([c, c + 1], [r, r], color='black')
                if cell_value & 2:  # Down wall
                    ax.plot([c, c + 1], [r + 1, r + 1], color='black')
                if cell_value & 4:  # Left wall
                    ax.plot([c, c], [r, r + 1], color='black')
                if cell_value & 8:  # Right wall
                    ax.plot([c + 1, c + 1], [r, r + 1], color='black')
        
        # Mark the start, end, and current positions
        ax.plot(self.origin_coords[1] + 0.5, self.origin_coords[0] + 0.5, 'go', label='Start')  # Start position
        ax.plot(self.target_coords[1] + 0.5, self.target_coords[0] + 0.5, 'ro', label='End')  # End position
        ax.plot(self.current_state[1] + 0.5, self.current_state[0] + 0.5, 'bo', label='Current')  # Current position
        
        # Set the limits and show the plot
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        
        # Add legend outside the plot
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        plt.draw()  # Use draw instead of show to avoid blocking

class QLearner:
    def __init__(self, env: MazeEnvironment, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1):
        self.env = env
        self.alpha = alpha # Learning rate (used in Q-value update)
        self.gamma = gamma # Discount factor (used in Q-value update)
        self.epsilon = epsilon # Exploration threshold (used in random choice for explore or exploit)
        self.q_table = self.initialize_q_table()

    def initialize_q_table(self):
        """ 
        Initialize the Q-table with zero values for all state-action pairs. While the values
        start at zero, they will be updated by rewards and future rewards as the agent
        explores the environment.

        Feel free to experiment with different initialization methods, or if you modify the maze
        environment to include more information or possible actions, you may need to adjust this
        method accordingly.
        """
        rows, cols = len(self.env.maze_grid), len(self.env.maze_grid[0])
        q_table = {}
        for r in range(rows):
            for c in range(cols):
                q_table[(r, c)] = [0, 0, 0, 0]  # Up, Down, Left, Right
        return q_table

    def choose_action(self, state):
        """
        Choose an action based on the current state. This method implements an epsilon-greedy
        policy, where the agent chooses a random action with probability epsilon, and the best
        known action with probability 1-epsilon.
        """
        if np.random.rand() < self.epsilon: 
            return np.random.choice([0, 1, 2, 3])
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, next_state, reward, done):
        """
        Update the Q-value for the given state-action pair based on the observed reward and the
        maximum Q-value for the next state.

        The formula used for updating the Q-value is:
        Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s, a))

        where:
        Q - is the table
        s - is the current state (row)
        a - is the action taken (column)
        alpha - is the learning rate (higher alpha means reward is added more to the Q-value, lower alpha means reward is added less)
        reward - is the reward received for taking action a in state s
        gamma - is the discount factor (higher gamma means future rewards are valued more, lower gamma means future rewards are valued less)
        s' - is the next state
        a' - is the action taken in the next state

        The logic being:
        First get the maximum Q-value for the next state (s') by taking the maximum value in the row of the Q-table corresponding to s'
        Multiply this maximum Q-value by the discount factor (gamma) which represents the value of future rewards
        Add the reward received for taking action a in state s to this discounted future reward
        Then subtract the current Q-value for state s and action a from this sum
        Multiply this difference by the learning rate (alpha) which represents how much we want to update the Q-value
        Finally add this product to the current Q-value for state s and action a to get the updated Q-value
        """
        q_value = self.q_table[state][action]
        if done:
            new_q_value = q_value + self.alpha * (reward - q_value)
        else:
            next_q_value = max(self.q_table[next_state])
            new_q_value = q_value + self.alpha * (reward + self.gamma * next_q_value - q_value)
        self.q_table[state][action] = new_q_value

    def train(self, episodes):
        """
        Train the agent by running a number of episodes. In each episode, the agent starts at the
        origin and takes actions until it reaches the target or runs out of steps.
        """
        for _ in tqdm.tqdm(range(episodes)):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update_q_table(state, action, next_state, reward, done)
                state = next_state

    def test_run(self, gif_path="maze_run.gif"):
        """
        Run the agent in the environment and save the run as a GIF.
        Use the maze environment's render method to visualize the run.
        """
        state = self.env.reset()
        done = False
        steps = 0
        fig, ax = plt.subplots()
        frames = []  # List to store frames for the GIF
        while not done:
            action = np.argmax(self.q_table[state])  # Choose the best action
            next_state, reward, done, _ = self.env.step(action)
            self.env.render(ax)  # Render the current state
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
            state = next_state
            steps += 1
        plt.close(fig)
        imageio.mimsave(gif_path, frames, fps=int(len(frames)/5))  # Save frames as GIF
        print(f"Test run completed in {steps} steps. GIF saved to {gif_path}.")

    def visualize_rewards(self):
        rows, cols = len(self.env.maze_grid), len(self.env.maze_grid[0])
        rewards = np.zeros((rows, cols))
        for r in range(rows):
            for c in range(cols):
                rewards[r, c] = max(self.q_table[(r, c)])
        plt.imshow(rewards, cmap='hot', interpolation='nearest')
        plt.show()

    def visualize_maze(self):
        self.env.render()

    def visualize_q_table(self):
        rows, cols = len(self.env.maze_grid), len(self.env.maze_grid[0])
        for r in range(rows):
            for c in range(cols):
                print(self.q_table[(r, c)], end=' ')
            print()

    def visualize_policy(self, save_path="maze_policy.txt"):
        rows, cols = len(self.env.maze_grid), len(self.env.maze_grid[0])
        policy = ""
        for r in range(rows):
            for c in range(cols):
                action = np.argmax(self.q_table[(r, c)])
                if action == 0:
                    policy += 'L '
                elif action == 1:
                    policy += 'R '
                elif action == 2:
                    policy += 'U '
                elif action == 3:
                    policy += 'D '
            policy += '\n'

        # Write policy to file (overwrite if exists)
        with open(save_path, 'w') as f:
            f.write(policy)
        print(f"Policy saved to {save_path}.")
        print(policy)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a maze.")
    parser.add_argument("-r", "--rows", type=int, default=5, help="Number of rows in the maze.")
    parser.add_argument("-c", "--cols", type=int, default=5, help="Number of columns in the maze.")
    parser.add_argument("--rc", "--cr", type=int, default=0, help="Square size for the maze.")
    parser.add_argument("--print-dumb", action="store_true", help="Print the maze in dumb format.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes for training.")
    args = parser.parse_args()

    if args.rc != 0:
        args.rows = args.rc
        args.cols = args.rc

    maze_grid, origin_coords, target_coords = generate_maze(args.rows, args.cols)
    
    env = MazeEnvironment(maze_grid, origin_coords, target_coords)
    q_learner = QLearner(env)
    q_learner.train(episodes=args.episodes)
    q_learner.test_run()
    q_learner.visualize_rewards()
    q_learner.visualize_policy()

    if args.print_dumb:
        dumb_maze_output = convert_to_alpha_format(maze_grid, origin_coords, target_coords)
        print(dumb_maze_output)

