# Final Goal Count Feature

This document describes the new final goal count feature that allows you to record and export the most accurate goal counts for teams and players.

## Overview

The football analysis system now supports three types of goal detection:

1. **Regular Goals**: Basic goal detection from the PassCounter system
2. **Enhanced Goals**: Advanced goal detection using field keypoints from the GoalDetector system
3. **Final Goals**: The most accurate goal count, determined by manual input or the best available detection system

## Priority System

The final goal count is determined using the following priority:

1. **Manual Goals** (highest priority) - If a manual goals CSV file is provided
2. **Enhanced Goals** - If the enhanced detection system detected goals
3. **Regular Goals** (fallback) - From the basic detection system

## Usage

### Creating a Manual Goals Template

To create a template CSV file for manual goal entry:

```bash
python main.py --create-goals-template goals_config.csv
```

This creates a CSV file with the following structure:

```csv
team,player_id,frame_num,comment
1,5,1200,Example goal for team 1 by player 5 at frame 1200
2,,3600,Example goal for team 2 by unknown player at frame 3600
1,10,,Example goal for team 1 by player 10 at unknown frame
```

### Manual Goals CSV Format

- **team**: Team number (1 or 2) - Required
- **player_id**: ID of the player who scored - Optional (leave empty for unknown player)
- **frame_num**: Frame number when goal occurred - Optional (for reference only)
- **comment**: Description of the goal - Optional

### Using Manual Goals

To process a video with manual goals:

```bash
python main.py --input video.mp4 --goals-config goals_config.csv
```

## Output Files

The system generates two comprehensive CSV files with all statistics including final goals:

### 1. Team Statistics File

- `{video_name}_team_stats.csv` - Contains all team statistics including final goals

### 2. Player Statistics File

- `{video_name}_player_stats.csv` - Contains all player statistics including final goals

### Example Output

**Team Statistics CSV:**

```csv
team,passes,goals,enhanced_goals,final_goals,tackles,interceptions
1,45,1,2,2,8,3
2,38,2,1,1,6,4
```

**Player Statistics CSV:**

```csv
player_id,jersey_number,team,passes,goals,enhanced_goals,final_goals,tackles,interceptions
5,5,1,12,1,1,1,2,1
10,10,1,8,0,1,1,1,0
7,7,2,15,2,0,0,3,2
```

## Benefits

1. **Accuracy**: Manual goals provide the most accurate count when automated detection fails
2. **Flexibility**: Falls back to enhanced or regular detection when manual goals aren't provided
3. **Comprehensive**: All statistics are consolidated into just two CSV files with final goal counts
4. **Comparison**: Can compare regular, enhanced, and final goal counts in the same file

## Example Workflow

1. Process video initially to see what goals were detected:

   ```bash
   python main.py --input match.mp4
   ```

2. Review the enhanced and regular goal detection results

3. Create a manual goals file if needed:

   ```bash
   python main.py --create-goals-template match_goals.csv
   ```

4. Edit the CSV file to add the correct goals

5. Reprocess with manual goals:

   ```bash
   python main.py --input match.mp4 --goals-config match_goals.csv
   ```

6. Use the `final_goals` column in the CSV files for the most accurate statistics

This ensures you always have the most accurate goal count regardless of the detection system's performance, all consolidated in just two comprehensive CSV files.
