# LunarLander-v2 Deep Q-Network (DQN) 🚀

This repository contains a PyTorch implementation of a **Deep Q-Network (DQN)** agent designed to solve the **LunarLander-v2** environment from OpenAI Gymnasium.

## Project Description

The goal of this project is to train an autonomous agent to safely land a spacecraft on a designated landing pad on the Moon. The landing pad is always at coordinates (0,0). 

### The Challenge
* **State Space**: 8-dimensional vector (x & y coordinates, x & y velocities, angle, angular velocity, and two booleans for leg contact).
* **Action Space**: 4 discrete actions (0: Do nothing, 1: Fire left engine, 2: Fire main engine, 3: Fire right engine).
* **Success Criteria**: The environment is considered "solved" when the agent achieves an average score of **+220 points** over 100 consecutive episodes.

## Model Architecture

The agent utilizes a Deep Q-Network with the following structure:
- **Input Layer**: 8 nodes (state size).
- **Hidden Layer 1**: 64 nodes with **ReLU** activation.
- **Hidden Layer 2**: 64 nodes with **ReLU** activation.
- **Output Layer**: 4 nodes (one for each action).

### Key Features:
* **Experience Replay**: A buffer of size 100,000 is used to store and sample transitions, reducing correlation between consecutive experiences.
* **Fixed Q-Targets**: A separate target network is used to provide stable Q-value targets during training.
* **Soft Updates**: The target network is updated slowly using a factor $\tau$ to ensure training stability.

## Hyperparameters

The following parameters were used to achieve optimal performance:

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| `Buffer Size` | 100,000 | Max capacity of the replay memory |
| `Batch Size` | 64 | Number of samples per training step |
| `Gamma ` | 0.99 | Discount factor for future rewards |
| `Tau ` | 1e-3 | Soft update interpolation parameter |
| `Learning Rate` | 5e-4 | Step size for the Adam optimizer |
| `Update Every` | 4 | Step frequency for network updates |
| `Epsilon Start` | 1.0 | Initial exploration rate |
| `Epsilon End` | 0.01 | Minimum exploration rate |
| `Epsilon Decay` | 0.995 | Rate at which epsilon decreases |

## Installation & Setup

To run this project locally, ensure you have the following dependencies installed:

### 1. System Requirements (Box2D)
```bash
sudo apt-get update
sudo apt-get install -y swig cmake
```
## 2. Python Environment

**Shell / Bash**

```bash
pip install gymnasium[box2d] torch numpy matplotlib stable-baselines3 huggingface_sb3
```

## Results

The agent successfully learned to land safely, showing a steady increase in cumulative rewards.

- **Solved in:** ~763 episodes (Average score > 220)
- **Performance:** Stable flight and precise landing with minimal fuel consumption

### Score Plot
The notebook generates a plot showing score variation across episodes, illustrating the learning curve as the agent transitions from random exploration to goal-oriented behavior.

---

## Acknowledgments

- **Gymnasium** – Standardized reinforcement learning environment  
- **Hugging Face Deep RL Course** – Foundational tools and requirements  
- **PyTorch** – Deep learning framework

