import random
import numpy as np
import marshal
import copy
from collections import deque


def generate_room(dim=(13, 13), p_change_directions=0.35, num_steps=25, num_boxes=3, tries=4, second_player=False, num_teleporters=0):
    """
    Generates a Sokoban room, represented by an integer matrix. The elements are encoded as follows:
    wall = 0
    empty space = 1
    box target = 2
    box not on target = 3
    box on target = 4
    player = 5
    teleporter = 7

    :param dim:
    :param p_change_directions:
    :param num_steps:
    :param num_boxes:
    :param tries:
    :param second_player:
    :param num_teleporters: Number of teleporter pairs to add
    :return: Numpy 2d Array and teleporter_pairs list
    """
    room_state = np.zeros(shape=dim)
    room_structure = np.zeros(shape=dim)
    teleporter_pairs = []

    # Some times rooms with a score == 0 are the only possibility.
    # In these case, we try another model.
    for t in range(tries):
        room = room_topology_generation(dim, p_change_directions, num_steps)
        room = place_boxes_and_player(room, num_boxes=num_boxes, second_player=second_player)

        # Room fixed represents all not movable parts of the room
        room_structure = np.copy(room)
        room_structure[room_structure == 5] = 1

        # Add teleporters to room structure BEFORE reverse playing
        # This ensures the reverse-playing algorithm validates solvability with teleporters
        teleporter_pairs = []
        if num_teleporters > 0:
            room_structure, teleporter_pairs = place_teleporters(room_structure, num_teleporters)

        # Room structure represents the current state of the room including movable parts
        room_state = room.copy()
        room_state[room_state == 2] = 4
        
        # Update room_state to include teleporters
        for pair in teleporter_pairs:
            for pos in pair:
                if room_state[pos[0], pos[1]] == 1:  # Only if it's empty floor
                    room_state[pos[0], pos[1]] = 7

        # Run reverse playing with teleporter awareness
        room_state, score, box_mapping = reverse_playing(room_state, room_structure, teleporter_pairs)
        room_state[room_state == 3] = 4

        if score > 0:
            break

    if score == 0:
        raise RuntimeWarning('Generated Model with score == 0')
    
    # Randomly move the player to make the level more challenging
    # and fix the issue where the player is always adjacent to a box after reverse_playing
    room_state = add_random_player_movement(
        room_state, 
        room_structure, 
        teleporter_pairs, 
        move_probability=0.7,  # 70% chance to move player
        continue_probability=0.6,  # 60% chance to continue moving
        max_steps=3  # Move up to 3 steps
    )

    return room_structure, room_state, box_mapping, teleporter_pairs


def room_topology_generation(dim=(10, 10), p_change_directions=0.35, num_steps=15):
    """
    Generate a room topology, which consits of empty floors and walls.

    :param dim:
    :param p_change_directions:
    :param num_steps:
    :return:
    """
    dim_x, dim_y = dim

    # The ones in the mask represent all fields which will be set to floors
    # during the random walk. The centered one will be placed over the current
    # position of the walk.
    masks = [
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0]
        ],
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0]
        ],
        [
            [0, 0, 0],
            [1, 1, 0],
            [0, 1, 0]
        ],
        [
            [0, 0, 0],
            [1, 1, 0],
            [1, 1, 0]
        ],
        [
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 0]
        ]
    ]

    # Possible directions during the walk
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    direction = random.sample(directions, 1)[0]

    # Starting position of random walk
    position = np.array([
        random.randint(1, dim_x - 1),
        random.randint(1, dim_y - 1)]
    )

    level = np.zeros(dim, dtype=int)

    for s in range(num_steps):

        # Change direction randomly
        if random.random() < p_change_directions:
            direction = random.sample(directions, 1)[0]

        # Update position
        position = position + direction
        position[0] = max(min(position[0], dim_x - 2), 1)
        position[1] = max(min(position[1], dim_y - 2), 1)

        # Apply mask
        mask = random.sample(masks, 1)[0]
        mask_start = position - 1
        level[mask_start[0]:mask_start[0] + 3, mask_start[1]:mask_start[1] + 3] += mask

    level[level > 0] = 1
    level[:, [0, dim_y - 1]] = 0
    level[[0, dim_x - 1], :] = 0

    return level


def place_boxes_and_player(room, num_boxes, second_player):
    """
    Places the player and the boxes into the floors in a room.

    :param room:
    :param num_boxes:
    :return:
    """
    # Get all available positions
    possible_positions = np.where(room == 1)
    num_possible_positions = possible_positions[0].shape[0]
    num_players = 2 if second_player else 1

    if num_possible_positions <= num_boxes + num_players:
        raise RuntimeError('Not enough free spots (#{}) to place {} player and {} boxes.'.format(
            num_possible_positions,
            num_players,
            num_boxes)
        )

    # Place player(s)
    ind = np.random.randint(num_possible_positions)
    player_position = possible_positions[0][ind], possible_positions[1][ind]
    room[player_position] = 5

    if second_player:
        ind = np.random.randint(num_possible_positions)
        player_position = possible_positions[0][ind], possible_positions[1][ind]
        room[player_position] = 5

    # Place boxes
    for n in range(num_boxes):
        possible_positions = np.where(room == 1)
        num_possible_positions = possible_positions[0].shape[0]

        ind = np.random.randint(num_possible_positions)
        box_position = possible_positions[0][ind], possible_positions[1][ind]
        room[box_position] = 2

    return room


def place_teleporters(room_structure, num_teleporters):
    """
    Places teleporter pairs into the room structure.
    Teleporters are placed on empty floor tiles (value 1) only.
    
    :param room_structure: The fixed room structure
    :param num_teleporters: Number of teleporter pairs to place
    :return: Updated room_structure and list of teleporter pairs [(pos1, pos2), ...]
    """
    teleporter_pairs = []
    
    for _ in range(num_teleporters):
        # Find empty spaces (not walls, boxes, targets, or player)
        possible_positions = np.argwhere(room_structure == 1)
        
        if len(possible_positions) < 2:
            print(f"[Room Generation] Warning: Not enough space to place teleporter pair")
            continue
        
        # Randomly select two positions for the teleporter pair
        indices = np.random.choice(len(possible_positions), size=2, replace=False)
        pos1 = tuple(possible_positions[indices[0]])
        pos2 = tuple(possible_positions[indices[1]])
        
        # Add the teleporter pair
        teleporter_pairs.append((pos1, pos2))
        
        # Update room_structure with teleporter marker (7)
        room_structure[pos1[0], pos1[1]] = 7
        room_structure[pos2[0], pos2[1]] = 7
    
    return room_structure, teleporter_pairs


# Global variables used for reverse playing.
explored_states = set()
num_boxes = 0
best_room_score = -1
best_room = None
best_box_mapping = None


def reverse_playing(room_state, room_structure, teleporter_pairs=None, search_depth=100):
    """
    This function plays Sokoban reverse in a way, such that the player can
    move and pull boxes.
    It ensures a solvable level with all boxes not being placed on a box target.
    Now supports teleporter mechanics for solvability validation.
    
    :param room_state:
    :param room_structure:
    :param teleporter_pairs: List of teleporter pairs [(pos1, pos2), ...]
    :param search_depth:
    :return: 2d array
    """
    global explored_states, num_boxes, best_room_score, best_room, best_box_mapping

    if teleporter_pairs is None:
        teleporter_pairs = []

    # Box_Mapping is used to calculate the box displacement for every box
    box_mapping = {}
    box_locations = np.where(room_structure == 2)
    num_boxes = len(box_locations[0])
    for l in range(num_boxes):
        box = (box_locations[0][l], box_locations[1][l])
        box_mapping[box] = box

    # explored_states globally stores the best room state and score found during search
    explored_states = set()
    best_room_score = -1
    best_box_mapping = box_mapping
    depth_first_search(room_state, room_structure, box_mapping, teleporter_pairs, 
                      box_swaps=0, last_pull=(-1, -1), ttl=300)

    return best_room, best_room_score, best_box_mapping


def depth_first_search(room_state, room_structure, box_mapping, teleporter_pairs, box_swaps=0, last_pull=(-1, -1), ttl=300):
    """
    Searches through all possible states of the room.
    This is a recursive function, which stops if the tll is reduced to 0 or
    over 1.000.000 states have been explored.
    Now includes teleporter mechanics in the search.
    
    :param room_state:
    :param room_structure:
    :param box_mapping:
    :param teleporter_pairs: List of teleporter pairs
    :param box_swaps:
    :param last_pull:
    :param ttl:
    :return:
    """
    global explored_states, num_boxes, best_room_score, best_room, best_box_mapping

    ttl -= 1
    if ttl <= 0 or len(explored_states) >= 300000:
        return

    state_tohash = marshal.dumps(room_state)

    # Only search this state, if it not yet has been explored
    if not (state_tohash in explored_states):

        # Add current state and its score to explored states
        room_score = box_swaps * box_displacement_score(box_mapping)
        if np.where(room_state == 2)[0].shape[0] != num_boxes:
            room_score = 0

        if room_score > best_room_score:
            best_room = room_state
            best_room_score = room_score
            best_box_mapping = box_mapping

        explored_states.add(state_tohash)

        for action in ACTION_LOOKUP.keys():
            # The state and box mapping  need to be copied to ensure
            # every action start from a similar state.
            room_state_next = room_state.copy()
            box_mapping_next = box_mapping.copy()

            room_state_next, box_mapping_next, last_pull_next = \
                reverse_move(room_state_next, room_structure, box_mapping_next, last_pull, action, teleporter_pairs)

            box_swaps_next = box_swaps
            if last_pull_next != last_pull:
                box_swaps_next += 1

            depth_first_search(room_state_next, room_structure,
                               box_mapping_next, teleporter_pairs, box_swaps_next,
                               last_pull, ttl)


def reverse_move(room_state, room_structure, box_mapping, last_pull, action, teleporter_pairs=None):
    """
    Perform reverse action. Where all actions in the range [0, 3] correspond to
    push actions and the ones greater 3 are simple move actions.
    Now includes teleporter mechanics.
    
    :param room_state:
    :param room_structure:
    :param box_mapping:
    :param last_pull:
    :param action:
    :param teleporter_pairs: List of teleporter pairs [(pos1, pos2), ...]
    :return:
    """
    if teleporter_pairs is None:
        teleporter_pairs = []
    
    player_position = np.where(room_state == 5)
    
    # Safety check: if no player found, return unchanged state
    if player_position[0].shape[0] == 0:
        return room_state, box_mapping, last_pull
    
    player_position = np.array([player_position[0][0], player_position[1][0]])

    change = CHANGE_COORDINATES[action % 4]
    next_position = player_position + change
    
    # Bounds check
    if (next_position[0] < 0 or next_position[0] >= room_state.shape[0] or
        next_position[1] < 0 or next_position[1] >= room_state.shape[1]):
        return room_state, box_mapping, last_pull

    # Check if next position is an empty floor, an empty box target, or a teleporter
    if room_state[next_position[0], next_position[1]] in [1, 2, 7]:

        # Move player, independent of pull or move action.
        room_state[player_position[0], player_position[1]] = room_structure[player_position[0], player_position[1]]
        room_state[next_position[0], next_position[1]] = 5
        
        # Track whether teleportation occurred
        did_teleport = False

        # Check if player landed on a teleporter and handle teleportation
        if room_structure[next_position[0], next_position[1]] == 7:
            # Find the paired teleporter
            teleport_dest = _find_teleporter_destination(tuple(next_position), teleporter_pairs)
            if teleport_dest is not None:
                # Check if destination is not blocked by a box
                if room_state[teleport_dest[0], teleport_dest[1]] not in [3, 4]:
                    # Perform teleportation
                    room_state[next_position[0], next_position[1]] = room_structure[next_position[0], next_position[1]]
                    room_state[teleport_dest[0], teleport_dest[1]] = 5
                    did_teleport = True
                    # Note: We don't update next_position because box pulling 
                    # should not occur when teleporting

        # In addition try to pull a box if the action is a pull action
        # IMPORTANT: Box pulling only happens if player did NOT teleport
        # (You cannot pull a box through a teleporter in reverse-playing)
        if action < 4 and not did_teleport:
            possible_box_location = change[0] * -1, change[1] * -1
            possible_box_location += player_position
            
            # Bounds check for box location
            if (possible_box_location[0] < 0 or possible_box_location[0] >= room_state.shape[0] or
                possible_box_location[1] < 0 or possible_box_location[1] >= room_state.shape[1]):
                return room_state, box_mapping, last_pull

            if room_state[possible_box_location[0], possible_box_location[1]] in [3, 4]:
                # Perform pull of the adjacent box
                room_state[player_position[0], player_position[1]] = 3
                room_state[possible_box_location[0], possible_box_location[1]] = room_structure[
                    possible_box_location[0], possible_box_location[1]]

                # Update the box mapping
                for k in box_mapping.keys():
                    if box_mapping[k] == (possible_box_location[0], possible_box_location[1]):
                        box_mapping[k] = (player_position[0], player_position[1])
                        last_pull = k

    return room_state, box_mapping, last_pull


def _find_teleporter_destination(from_position, teleporter_pairs):
    """
    Helper function to find the destination of a teleporter.
    
    :param from_position: tuple (row, col) of current teleporter
    :param teleporter_pairs: List of teleporter pairs
    :return: destination tuple or None
    """
    for pair in teleporter_pairs:
        if pair[0] == from_position:
            return pair[1]
        elif pair[1] == from_position:
            return pair[0]
    return None


def box_displacement_score(box_mapping):
    """
    Calculates the sum of all Manhattan distances, between the boxes
    and their origin box targets.
    :param box_mapping:
    :return:
    """
    score = 0
    
    for box_target in box_mapping.keys():
        box_location = np.array(box_mapping[box_target])
        box_target = np.array(box_target)
        dist = np.sum(np.abs(box_location - box_target))
        score += dist

    return score


TYPE_LOOKUP = {
    0: 'wall',
    1: 'empty space',
    2: 'box target',
    3: 'box on target',
    4: 'box not on target',
    5: 'player'
}

ACTION_LOOKUP = {
    0: 'push up',
    1: 'push down',
    2: 'push left',
    3: 'push right',
    4: 'move up',
    5: 'move down',
    6: 'move left',
    7: 'move right',
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


def add_random_player_movement(room_state, room_structure, teleporter_pairs=None, 
                                move_probability=0.5, continue_probability=0.5, max_steps=3):
    """
    Randomly move the player after reverse_playing to make the level more challenging,
    also fix the problem that in generated map, the player is always adjacent to the box.
    
    This version supports teleporter mechanics:
    - Player can move through teleporters
    - Teleportation happens automatically when stepping on a teleporter
    
    Parameters:
        room_state (np.ndarray): Current state of the room
        room_structure (np.ndarray): Fixed structure of the room
        teleporter_pairs (list): List of teleporter pairs [(pos1, pos2), ...]
        move_probability (float): Probability of moving the player at all (0.0-1.0)
        continue_probability (float): Probability of continuing to move after each step (0.0-1.0)
        max_steps (int): Maximum number of steps the player can move (1-3)
    
    Returns:
        np.ndarray: Updated room state with randomly moved player
    """
    if teleporter_pairs is None:
        teleporter_pairs = []
    
    # Check if we should move the player at all
    if random.random() > move_probability:
        return room_state
    
    # Find player position
    player_pos = np.where(room_state == 5)
    if player_pos[0].shape[0] == 0:
        return room_state  # No player found
    player_pos = np.array([player_pos[0][0], player_pos[1][0]])
    
    # Keep track of previous positions to avoid moving back
    previous_positions = [tuple(player_pos)]
    
    # Make 1-3 random moves
    steps_taken = 0
    while steps_taken < max_steps:
        # Get all valid moves (can't move into walls or boxes)
        valid_moves = []
        for action in range(4):  # 0: up, 1: down, 2: left, 3: right
            change = CHANGE_COORDINATES[action]
            next_pos = player_pos + change
            
            # Bounds check
            if (next_pos[0] < 0 or next_pos[0] >= room_state.shape[0] or
                next_pos[1] < 0 or next_pos[1] >= room_state.shape[1]):
                continue
            
            # Check if next position is valid (empty space, target, or teleporter) 
            # and not a previous position
            if (room_state[next_pos[0], next_pos[1]] in [1, 2, 7] and 
                tuple(next_pos) not in previous_positions):
                valid_moves.append((action, next_pos))
        
        # If no valid moves, break
        if not valid_moves:
            break
        
        # Choose a random valid move
        chosen_action, next_pos = random.choice(valid_moves)
        
        # Move player
        room_state[player_pos[0], player_pos[1]] = room_structure[player_pos[0], player_pos[1]]
        room_state[next_pos[0], next_pos[1]] = 5
        
        # Check if player landed on a teleporter
        final_pos = next_pos
        if room_structure[next_pos[0], next_pos[1]] == 7:
            # Find teleporter destination
            teleport_dest = _find_teleporter_destination(tuple(next_pos), teleporter_pairs)
            if teleport_dest is not None:
                # Check if destination is not blocked by a box
                if room_state[teleport_dest[0], teleport_dest[1]] not in [3, 4]:
                    # Perform teleportation
                    room_state[next_pos[0], next_pos[1]] = room_structure[next_pos[0], next_pos[1]]
                    room_state[teleport_dest[0], teleport_dest[1]] = 5
                    final_pos = np.array(teleport_dest)
        
        # Update player position and track previous position
        player_pos = final_pos
        previous_positions.append(tuple(player_pos))
        
        steps_taken += 1
        
        # Decide whether to continue moving
        if steps_taken >= max_steps or random.random() > continue_probability:
            break
    
    return room_state


def get_shortest_action_path(room_fixed, room_state, teleporter_pairs=None, MAX_DEPTH=100):
    """
    Get the shortest action path to push all boxes to the target spots.
    Use BFS to find the shortest path.
    
    This version supports teleporter mechanics:
    - Player automatically teleports when stepping on a teleporter
    - Teleportation is blocked if destination has a box
    
    NOTE: Currently only supports one player, only one shortest solution
    
    Parameters:
        room_state (np.ndarray): The state of the room
            - 0: wall
            - 1: empty space
            - 2: box target
            - 3: box on target
            - 4: box not on target
            - 5: player
            - 7: teleporter
        room_fixed (np.ndarray): The fixed part of the room
            - 0: wall
            - 1: empty space
            - 2: box target
            - 7: teleporter
        teleporter_pairs (list): List of teleporter pairs [(pos1, pos2), ...]
        MAX_DEPTH (int): The maximum depth of the search
    
    Returns:
        action_sequence (list): The action sequence to push all boxes to the target spots
    """
    if teleporter_pairs is None:
        teleporter_pairs = []
    
    # BFS queue stores (room_state, path)
    queue = deque([(copy.deepcopy(room_state), [])])
    explored_states = set()
    
    # Possible moves: up, down, left, right
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # Use push actions (1-4) - these handle both pushing and moving
    # In standard Sokoban, push actions automatically move if no box present
    actions = [1, 2, 3, 4]  # push/move up, down, left, right
    
    while queue:
        current_room_state, path = queue.popleft()
        if len(path) > MAX_DEPTH:
            return []  # No solution found
        
        # Reduce the search space by checking if the state has been explored
        state_tohash = marshal.dumps(current_room_state)
        if state_tohash in explored_states:
            continue
        explored_states.add(state_tohash)
        
        # Get information of the room
        player_pos_arr = np.argwhere(current_room_state == 5)
        if len(player_pos_arr) == 0:
            continue  # No player found, skip this state
        player_pos = tuple(player_pos_arr[0])
        
        boxes_on_target = set(map(tuple, np.argwhere((current_room_state == 3))))
        boxes_not_on_target = set(map(tuple, np.argwhere((current_room_state == 4))))
        boxes = boxes_on_target | boxes_not_on_target
        
        # Check if all boxes are on targets
        if not boxes_not_on_target:
            return path
        
        # Try each direction
        for move, action in zip(moves, actions):
            new_room_state = copy.deepcopy(current_room_state)
            new_player_pos = (player_pos[0] + move[0], player_pos[1] + move[1])
            
            # Check if new player position is wall or out of bound
            if (new_player_pos[0] < 0 or new_player_pos[0] >= room_fixed.shape[0] or
                new_player_pos[1] < 0 or new_player_pos[1] >= room_fixed.shape[1] or
                room_fixed[new_player_pos] == 0):
                continue
            
            # If there's a box, check if we can push it
            if new_player_pos in boxes:
                box_pos = new_player_pos  # The original box position
                new_box_pos = (new_player_pos[0] + move[0], new_player_pos[1] + move[1])
                
                # Can't push if hitting wall or another box or out of bound
                if (new_box_pos[0] < 0 or new_box_pos[0] >= room_fixed.shape[0] or
                    new_box_pos[1] < 0 or new_box_pos[1] >= room_fixed.shape[1] or
                    room_fixed[new_box_pos] == 0 or new_box_pos in boxes):
                    continue
                
                # Move the box
                new_room_state[box_pos] = room_fixed[box_pos]
                if room_fixed[new_box_pos] == 2:
                    new_room_state[new_box_pos] = 3
                else:
                    new_room_state[new_box_pos] = 4
            
            # Player moves
            new_room_state[player_pos] = room_fixed[player_pos]
            new_room_state[new_player_pos] = 5
            
            # Check if player landed on a teleporter and handle teleportation
            final_player_pos = new_player_pos
            if room_fixed[new_player_pos] == 7:
                teleport_dest = _find_teleporter_destination(new_player_pos, teleporter_pairs)
                if teleport_dest is not None:
                    # Check if destination is not blocked by a box
                    boxes_in_new_state = (set(map(tuple, np.argwhere((new_room_state == 3)))) |
                                         set(map(tuple, np.argwhere((new_room_state == 4)))))
                    if teleport_dest not in boxes_in_new_state:
                        # Perform teleportation
                        new_room_state[new_player_pos] = room_fixed[new_player_pos]
                        new_room_state[teleport_dest] = 5
                        final_player_pos = teleport_dest
            
            queue.append((new_room_state, path + [action]))
    
    return []  # No solution found


def plot_animation(imgs):
    """
    Create an animation from a sequence of images.
    
    Parameters:
        imgs (list): List of RGB images (numpy arrays)
    
    Returns:
        animation.FuncAnimation: The matplotlib animation object
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
    except ImportError:
        print("Error: matplotlib is required for plot_animation. Install with: pip install matplotlib")
        return None
    
    if not imgs:
        print("Error: No images provided for animation")
        return None
    
    height, width = imgs[0].shape[:2]
    fig = plt.figure(figsize=(width/100, height/100), dpi=500)
    
    ax = fig.add_axes([0, 0, 1, 1])
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    
    im = ax.imshow(imgs[0])
    
    def init():
        im.set_data(imgs[0])
        return [im]
    
    def update(i):
        im.set_data(imgs[i])
        return [im]
    
    ani = animation.FuncAnimation(fig, update, frames=len(imgs), init_func=init, blit=True)
    return ani


def solve_sokoban(env, saved_animation_path=None):
    """
    Solve the given sokoban environment and optionally save the animation.
    Works with teleporter-enabled environments.
    
    Parameters:
        env: The Sokoban environment instance
        saved_animation_path (str): Optional path to save the animation
    
    Returns:
        tuple: (actions, imgs) - The action sequence and list of rendered images
    """
    # Get teleporter pairs from environment
    teleporter_pairs = getattr(env, 'teleporter_pairs', [])
    
    # Find the shortest path
    actions = get_shortest_action_path(env.room_fixed, env.room_state, teleporter_pairs)
    
    if not actions:
        print("No solution found!")
        return [], []
    
    print(f"Found solution with {len(actions)} actions: {actions}")
    
    # Render the solution
    imgs = []
    img_before_action = env.render('rgb_array')
    imgs.append(img_before_action)
    
    for i, action in enumerate(actions):
        env.step(action)
        img_after_action = env.render('rgb_array')
        imgs.append(img_after_action)
        print(f"Step {i+1}: Action {action}")
    
    # Optionally save animation
    if saved_animation_path:
        try:
            ani = plot_animation(imgs)
            if ani:
                ani.save(saved_animation_path)
                print(f"Animation saved to: {saved_animation_path}")
        except Exception as e:
            print(f"Warning: Could not save animation: {e}")
    
    return actions, imgs
