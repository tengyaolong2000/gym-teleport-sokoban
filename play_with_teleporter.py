#!/usr/bin/env python3
"""
Interactive Sokoban with Teleporters - Full Action Space!

Controls:
- Arrow Keys: Push actions (move and push boxes)
- WASD: Move only actions (won't push boxes)
- Space: No operation (skip turn)
- R: Reset game
- N: New random level
- Q: Quit
"""

import numpy as np
import imageio
from gym_sokoban.envs import SokobanEnv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches


class InteractiveSokoban:
    """Interactive Sokoban game with keyboard controls."""
    
    def __init__(self, dim_room=(10, 10), num_boxes=3, num_teleporters=2):
        """Initialize the game."""
        self.dim_room = dim_room
        self.num_boxes = num_boxes
        self.num_teleporters = num_teleporters
        
        # Create environment
        self.env = SokobanEnv(
            dim_room=dim_room,
            num_boxes=num_boxes,
            num_teleporters=num_teleporters,
            max_steps=200
        )
        
        # Initialize game
        self.reset_game()
        
        # Action mapping for arrow keys (push) and WASD (move only)
        self.push_action_map = {
            'up': 1,      # push up
            'down': 2,    # push down
            'left': 3,    # push left
            'right': 4    # push right
        }
        self.move_action_map = {
            'w': 5,       # move up (no push)
            's': 6,       # move down (no push)
            'a': 7,       # move left (no push)
            'd': 8        # move right (no push)
        }
        self.noop_action = 0  # no operation
        
        # Statistics
        self.move_count = 0
        self.teleport_count = 0
        self.last_action = "None"
        self.last_teleport = False
        
        # Setup matplotlib
        self.setup_display()
    
    def reset_game(self):
        """Reset the game to initial state."""
        self.obs = self.env.reset()
        self.done = False
        self.total_reward = 0
        self.move_count = 0
        self.teleport_count = 0
        self.last_action = "Reset"
        self.last_teleport = False
        print("\n" + "=" * 60)
        print("NEW GAME STARTED")
        print("=" * 60)
        self.print_game_info()
    
    def print_game_info(self):
        """Print current game information."""
        print(f"\nRoom Size: {self.dim_room}")
        print(f"Boxes: {self.num_boxes}")
        print(f"Teleporter Pairs: {len(self.env.teleporter_pairs)}")
        for i, (pos1, pos2) in enumerate(self.env.teleporter_pairs, 1):
            print(f"  Pair {i}: {pos1} ↔ {pos2}")
        print(f"\nPlayer Position: {tuple(self.env.player_position)}")
        print(f"Boxes on Target: {self.env.boxes_on_target}/{self.num_boxes}")
        print()
    
    def setup_display(self):
        """Setup matplotlib display."""
        plt.ion()  # Interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.manager.set_window_title('Sokoban with Teleporters - Use Arrow Keys!')
        
        # Connect keyboard event
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Initial render
        self.update_display()
    
    def update_display(self):
        """Update the game display."""
        self.ax.clear()
        
        # Display game
        self.ax.imshow(self.obs)
        self.ax.axis('off')
        
        # Add game info as text
        info_text = f"Moves: {self.move_count} | Teleports: {self.teleport_count} | "
        info_text += f"Boxes on Target: {self.env.boxes_on_target}/{self.num_boxes}\n"
        info_text += f"Last Action: {self.last_action}"
        
        if self.last_teleport:
            info_text += " ✨ TELEPORTED! ✨"
        
        if self.done:
            if self.env._check_if_all_boxes_on_target():
                info_text += "\n🎉 PUZZLE SOLVED! 🎉 Press R to play again"
            else:
                info_text += "\n⏰ Time's up! Press R to try again"
        
        self.ax.set_title(info_text, fontsize=12, pad=20, fontweight='bold')
        
        # Add controls at bottom
        controls = "Controls: ↑↓←→ = Push | WASD = Move Only | Space = No Op | R = Reset | N = New Level | Q = Quit"
        self.fig.text(0.5, 0.02, controls, ha='center', fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.draw()
        plt.pause(0.01)
    
    def on_key_press(self, event):
        """Handle keyboard input."""
        if self.done and event.key not in ['r', 'n', 'q']:
            print("Game over! Press R to reset or N for new level.")
            return
        
        action = None
        action_name = None
        
        # Map keys to actions
        # Arrow keys - Push actions
        if event.key == 'up':
            action = self.push_action_map['up']
            action_name = 'Push Up'
        elif event.key == 'down':
            action = self.push_action_map['down']
            action_name = 'Push Down'
        elif event.key == 'left':
            action = self.push_action_map['left']
            action_name = 'Push Left'
        elif event.key == 'right':
            action = self.push_action_map['right']
            action_name = 'Push Right'
        # WASD keys - Move only actions (no push)
        elif event.key == 'w':
            action = self.move_action_map['w']
            action_name = 'Move Up'
        elif event.key == 's':
            action = self.move_action_map['s']
            action_name = 'Move Down'
        elif event.key == 'a':
            action = self.move_action_map['a']
            action_name = 'Move Left'
        elif event.key == 'd':
            action = self.move_action_map['d']
            action_name = 'Move Right'
        # Space bar - No operation
        elif event.key == ' ':
            action = self.noop_action
            action_name = 'No Op'
        # Control keys
        elif event.key == 'r':
            self.reset_game()
            self.update_display()
            return
        elif event.key == 'n':
            print("\n🎲 Generating new random level...")
            self.reset_game()
            self.update_display()
            return
        elif event.key == 'q':
            print("\n👋 Thanks for playing!")
            plt.close()
            return
        else:
            return
        
        # Execute action
        if action is not None and not self.done:
            # Store old position to detect teleportation
            old_pos = tuple(self.env.player_position)
            
            # Take action
            self.obs, reward, self.done, info = self.env.step(action)
            
            # Update statistics
            new_pos = tuple(self.env.player_position)
            moved = info.get('action.moved_player', False)
            
            if moved:
                self.move_count += 1
                self.last_action = action_name
                
                # Check if teleportation occurred (moved more than 1 tile)
                distance = abs(new_pos[0] - old_pos[0]) + abs(new_pos[1] - old_pos[1])
                if distance > 1:
                    self.teleport_count += 1
                    self.last_teleport = True
                    print(f"\n✨ TELEPORTED! {old_pos} → {new_pos} ✨")
                else:
                    self.last_teleport = False
                
                self.total_reward += reward
                
                # Print move info
                print(f"{action_name}: {old_pos} → {new_pos} | Reward: {reward:.2f}")
                
                if self.done:
                    if self.env._check_if_all_boxes_on_target():
                        print("\n" + "=" * 60)
                        print("🎉 CONGRATULATIONS! PUZZLE SOLVED! 🎉")
                        print("=" * 60)
                        print(f"Total Moves: {self.move_count}")
                        print(f"Teleports Used: {self.teleport_count}")
                        print(f"Total Reward: {self.total_reward:.2f}")
                        print("=" * 60)
                    else:
                        print("\n⏰ Maximum steps reached! Try again.")
            else:
                self.last_action = f"{action_name} (blocked)"
                self.last_teleport = False
                print(f"{action_name}: Blocked!")
            
            # Update display
            self.update_display()
    
    def run(self):
        """Run the interactive game."""
        print("\n" + "=" * 60)
        print("SOKOBAN WITH TELEPORTERS - FULL ACTION SPACE")
        print("=" * 60)
        print("\nControls:")
        print("  ↑ ↓ ← →  : Push actions (move and push boxes)")
        print("  W A S D  : Move only actions (won't push boxes)")
        print("  Space    : No operation (skip turn)")
        print("  R        : Reset current level")
        print("  N        : Generate new random level")
        print("  Q        : Quit game")
        print("\nAction Space:")
        print("  • Arrow keys use actions 1-4 (push)")
        print("  • WASD keys use actions 5-8 (move only)")
        print("  • Space bar uses action 0 (no-op)")
        print("\nTeleporter Info:")
        print("  • Purple/Blue tiles are teleporters")
        print("  • Step on a teleporter to instantly transport!")
        print("  • Teleporters are blocked if a box is on them")
        print("=" * 60)
        
        # Keep display open
        plt.show(block=True)


def main():
    """Main function."""
    print("\n🎮 Starting Interactive Sokoban with Teleporters...")
    
    # You can customize these parameters
    game = InteractiveSokoban(
        dim_room=(10, 10),      # Room size
        num_boxes=3,            # Number of boxes
        num_teleporters=2       # Number of teleporter pairs
    )
    
    game.run()


if __name__ == "__main__":
    main()

