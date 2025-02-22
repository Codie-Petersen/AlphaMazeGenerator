# AlphaMazeGenerator: A Critique of AlphaMaze's Efficiency

AlphaMazeGenerator is a project designed to critique the efficiency of using large language models (LLMs) for spatial reasoning tasks, specifically the [AlphaMaze project](https://github.com/janhq/visual-thinker). The [AlphaMaze interface](https://alphamaze.menlo.ai/) is limited to 5x5 mazes, which restricts its ability to handle more complex scenarios. This generator was created to test these limitations and explore more efficient methods for solving mazes. 

## Introduction

The AlphaMaze project aims to enhance visual reasoning in LLMs by having them solve mazes presented in text. However, the current implementation is constrained by its inability to handle mazes of arbitrary sizes. My attempt was to ensure that the random picker from the interface was not choosing random predetermined samples like some of the benchmark cheaters like OpenAI and others were doing. That succeeded, but only for 5x5 mazes. No other maze size works. So, I decided, instead of wasting the time I spent on makine the generator, just flush it out a little bit more with some old school reinforcement learning techniques.

## Key Features

- **Arbitrary Maze Sizes**: This generator can create mazes of any size (practical limits because of recursive functions), demonstrating the flexibility needed for real-world applications.
- **Q-Learning Algorithm**: A Q-Learning algorithm is integrated to navigate through the mazes to demonstrate a more straightforward and effective method compared to complex LLMs.
- **Critique of AlphaMaze**: By comparing the efficiency of Q-Learning with the computational resources required by AlphaMaze, this project argues that simpler methods are often more suitable for quantized spatial tasks.

## How It Works

1. **Maze Generation**: The project uses a recursive backtracking algorithm to generate mazes of specified sizes.
2. **Q-Learning Training**: The Q-Learning algorithm is trained on the generated maze to learn the optimal path from the origin to the target.
3. **Visualization Tools**: Includes tools for visualizing the maze, the Q-Learning policy, and the rewards distribution across the maze.

## Benefits

- **Efficiency**: Q-Learning provides a more efficient approach to solving mazes compared to complex visual reasoning tasks.
- **Flexibility**: Supports mazes of any size, allowing for more diverse and challenging scenarios.

## Conclusion

This project demonstrates that for quantized spatial tasks like maze solving, simpler algorithms like Q-Learning can be more efficient and effective than using large language models. It challenges the notion that complex models are necessary for spatial reasoning tasks. In my opinion, LLMs should have subsystems attached to them to accomplish these tasks and the LLMs can just utilize the results. LLMs should be trainined on something like the policy and what that policy means in relations to other semantic tasks/goals. Not the actual task of solving a maze or some other spatial task, which really isn't necessary.

## Future

I'll clean up some of the code and separate the conerns a little bit better later on. Ping me on LinkedIn if you end up modifying this significantly or using WITH LLMs, I'd be interested in what you are doing with it. And yes, this is very messy, not optimal code.
https://www.linkedin.com/in/petersen-synthetic-intel/

