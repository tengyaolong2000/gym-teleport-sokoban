# gym-sokoban 
[Sokoban](https://en.wikipedia.org/wiki/Sokoban) is Japanese for warehouse keeper and a traditional video game.
The game is a transportation puzzle, where the player has to push all boxes in the room on the storage locations/ targets.
The possibility of making irreversible mistakes makes these puzzles so challenging especially for [Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning) algorithms, which mostly lack the ability to think ahead.
<br/>The repository implements the game Sokoban based on the rules presented [DeepMind's]() paper [Imagination Augmented Agents for Deep Reinforcement Learning](https://papers.nips.cc/paper/7152-imagination-augmented-agents-for-deep-reinforcement-learning). 
The room generation is random and therefore, will allow to train Deep Neural Networks without overfitting on a set of predefined rooms.

## 🆕 NEW: Teleporter Functionality!

This fork extends the original gym-sokoban with **teleporter mechanics** - players can instantly transport between paired teleporter locations, adding a new strategic dimension to Sokoban puzzles!

### 🎮 Interactive Play with Teleporters
```bash
python3 play_with_teleporter.py
```
- **Arrow Keys**: Push actions (move and push boxes)
- **WASD**: Move only actions (won't push boxes)  
- **Space**: No operation (skip turn)
- **R**: Reset game
- **N**: New random level
- **Q**: Quit

### 🎬 Solve and Animate Puzzles
```bash
python3 solve_and_animate_simple.py
```
- Automatically generates and solves Sokoban puzzles
- Creates animated GIFs showing the complete solution
- Works with both regular and teleporter-enabled puzzles

| Example Game 1 | Example Game 2 | Example Game 3 |
| :---: | :---: | :---: 
| ![Game 1](/docs/Animations/solution_animation1.gif?raw=true) | ![Game 2](/docs/Animations/solution_animation2.gif?raw=true) | ![Game 3](/docs/Animations/solution_animation3.gif?raw=true) |

## 🚀 Teleporter Features

### ✨ Key Capabilities
- ✅ **Instant Transportation**: Players automatically teleport when stepping on a teleporter
- ✅ **Bidirectional**: Teleporter pairs work in both directions
- ✅ **Box Blocking**: Teleporters deactivate when a box is placed on them
- ✅ **Multiple Pairs**: Support for multiple independent teleporter pairs
- ✅ **Automatic & Manual Placement**: Can be randomly placed or manually configured
- ✅ **Full Rendering Support**: Works with all render modes (rgb_array, tiny, raw)

### 🎯 Quick Start with Teleporters

```python
from gym_sokoban.envs import SokobanEnv

# Create environment with 2 teleporter pairs
env = SokobanEnv(dim_room=(10, 10), num_boxes=3, num_teleporters=2)
obs = env.reset()

# Teleporter pairs are automatically placed
print(env.teleporter_pairs)
# Output: [((2, 3), (7, 8)), ((4, 5), (6, 7))]
```

### 🛠️ Manual Teleporter Placement

```python
# Create without teleporters
env = SokobanEnv(num_teleporters=0)
obs = env.reset()

# Manually set teleporter pairs
env.set_teleporter_pairs([
    ((2, 2), (8, 8)),  # Top-left to bottom-right
    ((5, 2), (5, 8))   # Left side to right side
])
```

## 1 Installation

### Via PIP
```bash
pip install gym-sokoban
```

### From Repository
```bash
git clone https://github.com/tengyaolong2000/gym-teleport-sokoban.git
cd gym-teleport-sokoban
pip install -e .
```




### 2.2 Actions
The game provides 9 actions to interact with the environment. 
Push and Move actions into the directions Up, Down, Left and Right.
The No Operation action is a void action, which does not change anything in the environment.
The mapping of the action numbers to the actual actions looks as follows

 | Action       | ID    | 
 | --------     | :---: | 
 | No Operation | 0     | 
 | Push Up      | 1     |  
 | Push Down    | 2     | 
 | Push Left    | 3     |   
 | Push Right   | 4     |   
 | Move Up      | 5     |
 | Move Down    | 6     |
 | Move Left    | 7     |
 | Move Right   | 8     |
 
**Move** simply moves if there is a free field in the direction, which means no blocking box or wall.

**Push** push tries to move an adjacent box if the next field behind the box is free.
This means no chain pushing of boxes is possible.
In case there is no box at the adjacent field, the push action is handled the same way as the move action into the same direction.

**Teleporter Mechanics**: When a player steps on a teleporter, they are instantly transported to the paired teleporter location. Teleporters become inactive when blocked by boxes.

### 2.3 Rewards
Finishing the game by pushing all on the targets gives a reward of 10 in the last step. 
Also pushing a box on or off a target gives a reward of 1 respectively of -1. 
In addition a reward of -0.1 is given for every step, this penalizes solutions with many steps.

| Reason                    | Reward |
| ------------------------- | ----:  |
| Perform Step              | -0.1   |
| Push Box on Target        |  1.0   |
| Push Box off Target       | -1.0   |
| Push all boxes on targets | 10.0   |

### 2.4 Level Generation
Every time a Sokoban environment is loaded or reset a new room is randomly generated.
The generation consists of 3 phases: Topology Generation, Placement of Targets and Players, and Reverse Playing.
#### 2.4.1 Topology Generation
To generate the basic topology of the room, consisting of walls and empty floor, is based on a random walk, which changes its direction at probability 0.35.
At every step centered at the current position, a pattern of fields is set to empty spaces.
The patterns used can be found in [Figure 2](#topologyMask).
<div style="padding:20%">
  <p align="center">
    <img src="/docs/masks.png?raw=true">
  </p>
  <p align="center" id="topologyMask">
    Figure 2: Masks for creating a topology
  </p>
</div>


#### 2.4.2 Placement of Elements
During this phase, the player including all n box targets are placed on randomly chosen empty spaces.

#### 2.4.3 Reverse Playing
This is the crucial phase to  ensure a solvable room.
Now Sokoban is played in a reverse fashion, where a player can move and pull boxes.
The goal of this phase is to find the room state, with the highest room score, with a [Depth First Search](https://en.wikipedia.org/wiki/Depth-first_search).
For every room explored during the search is a room score is calculated with the equation shown below.
The equation is a heuristic approach to evaluate the difficulty of the room.
BoxSwaps counts the number of times a player changes the box to pull.
BoxDisplacement is the [Manhattan Distance](https://en.wikipedia.org/wiki/Manhattan_distance) between a specific box and its origin box target. 
As long as at least one box is on a target the RoomScore is always 0.
<div style="padding:10%">
  <p align="center">
   <img src="https://latex.codecogs.com/svg.latex?\Large&space;RoomScore&space;=&space;BoxSwaps&space;\times&space;\sum_{i&space;\in&space;Boxes}_{BoxDisplacement_{i}}" title="Room Score" />
  </p>
</div>

### 2.5 Configuration
Sokoban has many different variations, such as: Room Size, Number of Boxes, Rendering Modes, or Rules.

#### 2.5.1 Rendering Modes
Besides the regular Sokoban rendering, each configuration can be rendered as TinyWorld, which has a pixel size equal to the grid size. 
To get an environment rendered as a tiny world just add `tiny_` in front of the rendering mode. E.g: `env.render('tiny_rgb_array', scale=scale_tiny)`. Scale allows to increase the size of the rendered tiny world observation. Using scale in combination with the rendering modes, `human` or `rgb_array`, does not influence the output size.
Available rendering modes are:



