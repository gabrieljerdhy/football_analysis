import csv
import os


def create_goals_template(output_path="goals_config.csv"):
    """
    Create a template CSV file for manual goal configuration.
    
    Args:
        output_path (str): Path to save the template file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['team', 'player_id', 'frame_num', 'comment']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        # Add some example rows
        writer.writerow({
            'team': 1,
            'player_id': 5,  # Optional, can be empty
            'frame_num': 1200,  # Optional, can be empty
            'comment': 'Example goal for team 1 by player 5 at frame 1200'
        })
        writer.writerow({
            'team': 2,
            'player_id': '',  # Unknown player
            'frame_num': 3600,
            'comment': 'Example goal for team 2 by unknown player at frame 3600'
        })
        writer.writerow({
            'team': 1,
            'player_id': 10,
            'frame_num': '',  # Unknown frame
            'comment': 'Example goal for team 1 by player 10 at unknown frame'
        })
    
    print(f"Goals template created at {output_path}")
    print("Edit this file to add manual goals, then use it with --goals-config option")    print("Edit this file to add manual goals, then use it with --goals-config option")