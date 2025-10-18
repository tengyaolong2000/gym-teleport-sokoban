#!/usr/bin/env python3
"""
Simple demo: Generate, solve, and animate a Sokoban puzzle
Works without gym dependency by using room_utils directly
"""

import sys
import numpy as np
sys.path.insert(0, '/Users/tengyaolong/Desktop/gym-sokoban')
sys.path.insert(0, '/Users/tengyaolong/Desktop/gym-sokoban/gym_sokoban/envs')

from room_utils import (
    generate_room, 
    get_shortest_action_path, 
    plot_animation,
    CHANGE_COORDINATES
)
from render_utils import room_to_rgb
import matplotlib.pyplot as plt

def simulate_actions(room_fixed, room_state, actions, teleporter_pairs=None):
    """
    Simulate actions and collect rendered frames.
    Now includes teleporter support!
    """
    if teleporter_pairs is None:
        teleporter_pairs = []
    
    imgs = []
    
    # Render initial state
    imgs.append(room_to_rgb(room_state, room_fixed))
    
    # Apply each action
    for action in actions:
        # Get player position
        player_pos = np.argwhere(room_state == 5)
        if len(player_pos) == 0:
            break
        player_pos = tuple(player_pos[0])
        
        # Get movement direction (actions 1-4 map to CHANGE_COORDINATES 0-3)
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_player_pos = (player_pos[0] + change[0], player_pos[1] + change[1])
        
        # Check if there's a box to push
        if room_state[new_player_pos] in [3, 4]:
            # There's a box, try to push it
            new_box_pos = (new_player_pos[0] + change[0], new_player_pos[1] + change[1])
            
            # Check if push is valid
            if room_state[new_box_pos] in [1, 2, 7]:
                # Move box
                room_state[new_box_pos] = 3 if room_fixed[new_box_pos] == 2 else 4
                room_state[new_player_pos] = room_fixed[new_player_pos]
        
        # Move player
        if room_state[new_player_pos] in [1, 2, 7]:
            room_state[player_pos] = room_fixed[player_pos]
            room_state[new_player_pos] = 5
            
            # ✅ NEW: Check if player landed on a teleporter and handle teleportation
            if room_fixed[new_player_pos] == 7:
                # Find the paired teleporter
                teleport_dest = None
                for pair in teleporter_pairs:
                    if pair[0] == new_player_pos:
                        teleport_dest = pair[1]
                        break
                    elif pair[1] == new_player_pos:
                        teleport_dest = pair[0]
                        break
                
                # Check if destination is not blocked by a box
                if teleport_dest is not None:
                    if room_state[teleport_dest] not in [3, 4]:
                        # Perform teleportation
                        room_state[new_player_pos] = room_fixed[new_player_pos]
                        room_state[teleport_dest] = 5
        
        # Render new state
        imgs.append(room_to_rgb(room_state, room_fixed))
    
    return imgs

def main():
    print("\n" + "=" * 70)
    print("SOKOBAN SOLVER & ANIMATION DEMO (Simple Version)")
    print("=" * 70)
    
    # Generate a room
    print("\n[Step 1] Generating Sokoban puzzle...")
    room_fixed, room_state, box_mapping, teleporter_pairs = generate_room(
        dim=(10, 10),
        num_boxes=2,
        num_teleporters=2,
        num_steps=35,  # Increased from 15 to create larger rooms
        tries=4
    )
    
    print(f"✓ Generated 8x8 room")
    print(f"  Boxes: 2")
    print(f"  Teleporter pairs: {len(teleporter_pairs)}")
    if teleporter_pairs:
        for i, pair in enumerate(teleporter_pairs, 1):
            print(f"    Pair {i}: {pair[0]} ↔ {pair[1]}")
    
    player_pos = np.argwhere(room_state == 5)
    if len(player_pos) > 0:
        print(f"  Player at: {tuple(player_pos[0])}")
    
    # Solve the puzzle
    print("\n[Step 2] Solving puzzle with BFS solver...")
    print("  (This may take a few seconds...)")
    
    actions = get_shortest_action_path(
        room_fixed,
        room_state.copy(),  # Copy so we don't modify original
        teleporter_pairs,
        MAX_DEPTH=100
    )
    
    if actions:
        print(f"✓ Solution found!")
        print(f"  Moves required: {len(actions)}")
        print(f"  Action sequence: {actions}")
        
        # Simulate and render solution
        print("\n[Step 3] Rendering solution frames...")
        imgs = simulate_actions(room_fixed, room_state.copy(), actions, teleporter_pairs)
        print(f"  Frames captured: {len(imgs)}")
        
        # Display key frames
        print("\n[Step 4] Creating visualization...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(imgs[0])
        axes[0].set_title('Initial State', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        mid_idx = len(imgs) // 2
        axes[1].imshow(imgs[mid_idx])
        axes[1].set_title(f'Middle (Step {mid_idx}/{len(imgs)-1})', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(imgs[-1])
        axes[2].set_title('Final State (Solved!)', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        output_path = '/Users/tengyaolong/Desktop/gym-sokoban/solution_steps.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Key frames saved to: solution_steps.png")
        plt.close()
        
        # Create and save animation
        print("\n[Step 5] Creating animation...")
        ani = plot_animation(imgs)
        
        if ani:
            gif_path = '/Users/tengyaolong/Desktop/gym-sokoban/solution_animation.gif'
            print(f"  Saving animation to: {gif_path}")
            try:
                ani.save(gif_path, writer='pillow', fps=2)
                print("✓ Animation saved successfully!")
            except Exception as e:
                print(f"⚠️  Could not save GIF: {e}")
                print("   Trying to display instead...")
                try:
                    plt.show()
                except:
                    pass
        
        print("\n" + "=" * 70)
        print("✓ DEMO COMPLETE!")
        print("=" * 70)
        print("\nFiles created:")
        print("  - solution_steps.png (shows initial, middle, and final states)")
        if ani:
            print("  - solution_animation.gif (full solution animation)")
        print("\nThe puzzle was:")
        print(f"  - Generated with teleporters: {len(teleporter_pairs) > 0}")
        print(f"  - Solved in {len(actions)} moves")
        print(f"  - Rendered as {len(imgs)} frames")
        
    else:
        print("✗ No solution found within search depth")
        print("  (Trying to display puzzle anyway...)")
        
        # Display the unsolved puzzle
        img = room_to_rgb(room_state, room_fixed)
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title('Generated Puzzle (Unsolved)', fontsize=14)
        plt.axis('off')
        plt.savefig('/Users/tengyaolong/Desktop/gym-sokoban/puzzle_unsolved.png')
        print("  Saved puzzle to: puzzle_unsolved.png")

if __name__ == "__main__":
    main()

