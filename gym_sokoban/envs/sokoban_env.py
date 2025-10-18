import gym
from gym.utils import seeding
from gym.spaces.discrete import Discrete
from gym.spaces import Box
from .room_utils import generate_room
from .render_utils import room_to_rgb, room_to_tiny_world_rgb
import numpy as np


class SokobanEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array', 'raw'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array', 'raw']
    }

    def __init__(self,
                 dim_room=(10, 10),
                 max_steps=120,
                 num_boxes=4,
                 num_gen_steps=None,
                 num_teleporters=0,
                 reset=True):

        # General Configuration
        self.dim_room = dim_room
        if num_gen_steps == None:
            self.num_gen_steps = int(1.7 * (dim_room[0] + dim_room[1]))
        else:
            self.num_gen_steps = num_gen_steps

        self.num_boxes = num_boxes
        self.num_teleporters = num_teleporters
        self.teleporter_pairs = []  # List of (from_pos, to_pos) tuples
        self.boxes_on_target = 0

        # Penalties and Rewards
        self.penalty_for_step = -0.1
        self.penalty_box_off_target = -1
        self.reward_box_on_target = 1
        self.reward_finished = 10
        self.reward_last = 0

        # Other Settings
        self.viewer = None
        self.max_steps = max_steps
        self.action_space = Discrete(len(ACTION_LOOKUP))
        screen_height, screen_width = (dim_room[0] * 16, dim_room[1] * 16)
        self.observation_space = Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        
        if reset:
            # Initialize Room
            _ = self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, observation_mode='rgb_array'):
        assert action in ACTION_LOOKUP
        assert observation_mode in ['rgb_array', 'tiny_rgb_array', 'raw']

        self.num_env_steps += 1

        self.new_box_position = None
        self.old_box_position = None

        moved_box = False

        if action == 0:
            moved_player = False

        # All push actions are in the range of [0, 3]
        elif action < 5:
            moved_player, moved_box = self._push(action)

        else:
            moved_player = self._move(action)

        self._calc_reward()
        
        done = self._check_if_done()

        # Convert the observation to RGB frame
        observation = self.render(mode=observation_mode)

        info = {
            "action.name": ACTION_LOOKUP[action],
            "action.moved_player": moved_player,
            "action.moved_box": moved_box,
        }
        if done:
            info["maxsteps_used"] = self._check_if_maxsteps()
            info["all_boxes_on_target"] = self._check_if_all_boxes_on_target()

        return observation, self.reward_last, done, info

    def _push(self, action):
        """
        Perform a push, if a box is adjacent in the right direction.
        If no box, can be pushed, try to move.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # No push, if the push would get the box out of the room's grid
        new_box_position = new_position + change
        if new_box_position[0] >= self.room_state.shape[0] \
                or new_box_position[1] >= self.room_state.shape[1]:
            return False, False


        can_push_box = self.room_state[new_position[0], new_position[1]] in [3, 4]
        can_push_box &= self.room_state[new_box_position[0], new_box_position[1]] in [1, 2, 7]
        if can_push_box:

            self.new_box_position = tuple(new_box_position)
            self.old_box_position = tuple(new_position)

            # Move Player
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            # Move Box
            box_type = 4
            if self.room_fixed[new_box_position[0], new_box_position[1]] == 2:
                box_type = 3
            self.room_state[new_box_position[0], new_box_position[1]] = box_type
            
            # Check if player landed on a teleporter after pushing and teleport them
            if self.room_fixed[new_position[0], new_position[1]] == 7:
                self._try_teleport(tuple(new_position))
            
            return True, True

        # Try to move if no box to push, available
        else:
            return self._move(action), False

    def _move(self, action):
        """
        Moves the player to the next field, if it is not occupied.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # Move player if the field in the moving direction is either
        # an empty field, an empty box target, or a teleporter.
        if self.room_state[new_position[0], new_position[1]] in [1, 2, 7]:
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            # Check if player stepped on a teleporter and teleport them
            if self.room_fixed[new_position[0], new_position[1]] == 7:
                teleported = self._try_teleport(tuple(new_position))
                if teleported:
                    return True

            return True

        return False

    def _try_teleport(self, from_position):
        """
        Attempts to teleport the player from the given position to its paired teleporter.
        Teleportation only occurs if the destination teleporter is not blocked by a box.
        :param from_position: tuple (row, col) of the teleporter the player is on
        :return: Boolean indicating whether teleportation occurred
        """
        # Find the paired teleporter
        destination = None
        for tp_pair in self.teleporter_pairs:
            if tp_pair[0] == from_position:
                destination = tp_pair[1]
                break
            elif tp_pair[1] == from_position:
                destination = tp_pair[0]
                break
        
        if destination is None:
            return False
        
        # Check if destination teleporter is blocked by a box
        if self._is_teleporter_blocked(destination):
            return False
        
        # Perform teleportation
        current_position = self.player_position.copy()
        self.player_position = np.array(destination)
        
        # Update room state
        self.room_state[current_position[0], current_position[1]] = \
            self.room_fixed[current_position[0], current_position[1]]
        self.room_state[destination[0], destination[1]] = 5
        
        return True
    
    def _is_teleporter_blocked(self, teleporter_position):
        """
        Checks if a teleporter is blocked by a box.
        :param teleporter_position: tuple (row, col) of the teleporter
        :return: Boolean indicating whether the teleporter is blocked
        """
        # A teleporter is blocked if there's a box on it (state 3 or 4)
        return self.room_state[teleporter_position[0], teleporter_position[1]] in [3, 4]

    def _calc_reward(self):
        """
        Calculate Reward Based on
        :return:
        """
        # Every step a small penalty is given, This ensures
        # that short solutions have a higher reward.
        self.reward_last = self.penalty_for_step

        # count boxes off or on the target
        empty_targets = self.room_state == 2
        player_on_target = (self.room_fixed == 2) & (self.room_state == 5)
        total_targets = empty_targets | player_on_target

        current_boxes_on_target = self.num_boxes - \
                                  np.where(total_targets)[0].shape[0]

        # Add the reward if a box is pushed on the target and give a
        # penalty if a box is pushed off the target.
        if current_boxes_on_target > self.boxes_on_target:
            self.reward_last += self.reward_box_on_target
        elif current_boxes_on_target < self.boxes_on_target:
            self.reward_last += self.penalty_box_off_target
        
        game_won = self._check_if_all_boxes_on_target()        
        if game_won:
            self.reward_last += self.reward_finished
        
        self.boxes_on_target = current_boxes_on_target

    def _check_if_done(self):
        # Check if the game is over either through reaching the maximum number
        # of available steps or by pushing all boxes on the targets.        
        return self._check_if_all_boxes_on_target() or self._check_if_maxsteps()

    def _check_if_all_boxes_on_target(self):
        empty_targets = self.room_state == 2
        player_hiding_target = (self.room_fixed == 2) & (self.room_state == 5)
        are_all_boxes_on_targets = np.where(empty_targets | player_hiding_target)[0].shape[0] == 0
        return are_all_boxes_on_targets

    def _check_if_maxsteps(self):
        return (self.max_steps == self.num_env_steps)

    def reset(self, second_player=False, render_mode='rgb_array'):
        try:
            # Generate room with teleporters already included and validated for solvability
            self.room_fixed, self.room_state, self.box_mapping, self.teleporter_pairs = generate_room(
                dim=self.dim_room,
                num_steps=self.num_gen_steps,
                num_boxes=self.num_boxes,
                second_player=second_player,
                num_teleporters=self.num_teleporters
            )
        except (RuntimeError, RuntimeWarning) as e:
            print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
            print("[SOKOBAN] Retry . . .")
            return self.reset(second_player=second_player, render_mode=render_mode)

        self.player_position = np.argwhere(self.room_state == 5)[0]
        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0
        
        # Teleporters are now already placed and validated during room generation
        # No need to add them separately anymore

        starting_observation = self.render(render_mode)
        return starting_observation

    def render(self, mode='human', close=None, scale=1):
        assert mode in RENDERING_MODES

        img = self.get_image(mode, scale)

        if 'rgb_array' in mode:
            return img

        elif 'human' in mode:
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

        elif 'raw' in mode:
            arr_walls = (self.room_fixed == 0).view(np.int8)
            arr_goals = (self.room_fixed == 2).view(np.int8)
            arr_boxes = ((self.room_state == 4) + (self.room_state == 3)).view(np.int8)
            arr_player = (self.room_state == 5).view(np.int8)
            arr_teleporters = (self.room_fixed == 7).view(np.int8)

            return arr_walls, arr_goals, arr_boxes, arr_player, arr_teleporters

        else:
            super(SokobanEnv, self).render(mode=mode)  # just raise an exception

    def get_image(self, mode, scale=1):
        
        if mode.startswith('tiny_'):
            img = room_to_tiny_world_rgb(self.room_state, self.room_fixed, scale=scale)
        else:
            img = room_to_rgb(self.room_state, self.room_fixed)

        return img

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def set_maxsteps(self, num_steps):
        self.max_steps = num_steps

    def get_action_lookup(self):
        return ACTION_LOOKUP

    def get_action_meanings(self):
        return ACTION_LOOKUP

    def set_teleporter_pairs(self, teleporter_pairs):
        """
        Set teleporter pairs for the environment.
        :param teleporter_pairs: List of tuples [(from_pos, to_pos), ...] where each position is (row, col)
        """
        self.teleporter_pairs = teleporter_pairs
        # Update room_fixed to mark teleporter positions
        for pair in teleporter_pairs:
            self.room_fixed[pair[0][0], pair[0][1]] = 7
            self.room_fixed[pair[1][0], pair[1][1]] = 7
            # If there's no box on the teleporter, update room_state as well
            if self.room_state[pair[0][0], pair[0][1]] not in [3, 4]:
                self.room_state[pair[0][0], pair[0][1]] = 7
            if self.room_state[pair[1][0], pair[1][1]] not in [3, 4]:
                self.room_state[pair[1][0], pair[1][1]] = 7
    
    def add_teleporters_to_room(self, num_teleporters=1):
        """
        Adds teleporter pairs to the current room.
        :param num_teleporters: Number of teleporter pairs to add
        """
        for _ in range(num_teleporters):
            # Find empty spaces (not walls, boxes, targets, or player)
            possible_positions = np.argwhere((self.room_state == 1) & (self.room_fixed == 1))
            
            if len(possible_positions) < 2:
                print(f"[SOKOBAN] Warning: Not enough space to place teleporter pair")
                continue
            
            # Randomly select two positions for the teleporter pair
            indices = np.random.choice(len(possible_positions), size=2, replace=False)
            pos1 = tuple(possible_positions[indices[0]])
            pos2 = tuple(possible_positions[indices[1]])
            
            # Add the teleporter pair
            self.teleporter_pairs.append((pos1, pos2))
            
            # Update room_fixed and room_state
            self.room_fixed[pos1[0], pos1[1]] = 7
            self.room_fixed[pos2[0], pos2[1]] = 7
            self.room_state[pos1[0], pos1[1]] = 7
            self.room_state[pos2[0], pos2[1]] = 7


ACTION_LOOKUP = {
    0: 'no operation',
    1: 'push up',
    2: 'push down',
    3: 'push left',
    4: 'push right',
    5: 'move up',
    6: 'move down',
    7: 'move left',
    8: 'move right',
}

# Moves are mapped to coordinate changes as follows
# 0: Move up
# 1: Move down
# 2: Move left
# 3: Move right
CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

RENDERING_MODES = ['rgb_array', 'human', 'tiny_rgb_array', 'tiny_human', 'raw']
