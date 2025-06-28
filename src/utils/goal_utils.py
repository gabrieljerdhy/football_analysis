import csv
import os


def load_manual_goals(goals_config_path):
    """
    Load manual goals from a CSV configuration file.

    Args:
        goals_config_path (str): Path to the CSV file with manual goal information

    Returns:
        list: List of goal dictionaries with keys: team, player_id, frame_num, comment
    """
    if not goals_config_path or not os.path.exists(goals_config_path):
        return []

    manual_goals = []

    try:
        with open(goals_config_path, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                # Skip empty rows or rows without team
                if not row.get("team"):
                    continue

                goal = {
                    "team": int(row["team"]),
                    "player_id": (
                        int(row["player_id"])
                        if row.get("player_id") and row["player_id"].strip()
                        else None
                    ),
                    "frame_num": (
                        int(row["frame_num"])
                        if row.get("frame_num") and row["frame_num"].strip()
                        else None
                    ),
                    "comment": row.get("comment", "").strip(),
                }

                manual_goals.append(goal)

        print(f"📋 Loaded {len(manual_goals)} manual goals from {goals_config_path}")
        for goal in manual_goals:
            player_info = f" by Player {goal['player_id']}" if goal["player_id"] else ""
            frame_info = f" at frame {goal['frame_num']}" if goal["frame_num"] else ""
            print(f"   - Team {goal['team']}{player_info}{frame_info}")

    except Exception as e:
        print(f"❌ Error loading manual goals from {goals_config_path}: {e}")
        return []

    return manual_goals


def create_goals_template(output_path="data/goals_config.csv"):
    """
    Create a template CSV file for manual goal configuration.

    Args:
        output_path (str): Path to save the template file
    """
    # Create directory if it doesn't exist
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )

    with open(output_path, "w", newline="") as csvfile:
        fieldnames = ["team", "player_id", "frame_num", "comment"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        # Add some example rows
        writer.writerow(
            {
                "team": 1,
                "player_id": 5,  # Optional, can be empty
                "frame_num": 1200,  # Optional, can be empty
                "comment": "Example goal for team 1 by player 5 at frame 1200",
            }
        )
        writer.writerow(
            {
                "team": 2,
                "player_id": "",  # Unknown player
                "frame_num": 3600,
                "comment": "Example goal for team 2 by unknown player at frame 3600",
            }
        )
        writer.writerow(
            {
                "team": 1,
                "player_id": 10,
                "frame_num": "",  # Unknown frame
                "comment": "Example goal for team 1 by player 10 at unknown frame",
            }
        )

    print(f"Goals template created at {output_path}")
    print("Edit this file to add manual goals, then use it with --goals-config option")


def calculate_final_goal_stats(
    pass_counter, enhanced_goal_stats, manual_goals, scoreboard_analyzer=None
):
    """
    Calculate final goal statistics based on priority system.
    The final goal count is determined by:
    1. Manual goals (highest priority)
    2. Scoreboard-extracted scores (if available and reliable)
    3. Enhanced goal detection
    4. Regular goal detection (lowest priority)

    Args:
        pass_counter: PassCounter instance with regular goal detection
        enhanced_goal_stats: Enhanced goal statistics from GoalDetector
        manual_goals: List of manual goals loaded from CSV
        scoreboard_analyzer: ScoreboardAnalyzer instance (optional)

    Returns:
        tuple: (final_team_goals, final_player_goals)
    """
    # Initialize final goal counts
    final_team_goals = {1: 0, 2: 0}
    final_player_goals = {}

    # Get scoreboard analysis results
    scoreboard_score = None
    if scoreboard_analyzer and scoreboard_analyzer.is_scoreboard_available():
        scoreboard_score = scoreboard_analyzer.get_final_score()

    print("📊 Calculating final goal statistics...")
    print(f"   Manual goals provided: {len(manual_goals) if manual_goals else 0}")
    if scoreboard_score:
        print(
            f"   Scoreboard score detected: {scoreboard_score['team1_score']}-{scoreboard_score['team2_score']} (confidence: {scoreboard_score['confidence']:.2f})"
        )
    else:
        print("   Scoreboard score: Not detected")
    print(
        f"   Enhanced goals detected: {enhanced_goal_stats.get('final_total_goals', 0) if enhanced_goal_stats else 0}"
    )
    print(f"   Regular goals detected: {sum(pass_counter.team_goals.values())}")

    # Priority system for determining final goal count
    # 1. Manual goals (highest priority)
    if manual_goals:
        print("📊 Using manual goals as final goal count")
        for goal in manual_goals:
            team = goal["team"]
            player_id = goal["player_id"]

            # Update team goals
            final_team_goals[team] = final_team_goals.get(team, 0) + 1

            # Update player goals if player is specified
            if player_id is not None:
                if player_id not in final_player_goals:
                    final_player_goals[player_id] = {"goals": 0, "team": team}
                final_player_goals[player_id]["goals"] += 1

    # 2. Scoreboard-extracted scores (if available and reliable)
    elif scoreboard_score and scoreboard_score.get("confidence", 0) >= 0.7:
        print("📊 Using scoreboard-extracted scores as final goal count")
        final_team_goals[1] = scoreboard_score["team1_score"]
        final_team_goals[2] = scoreboard_score["team2_score"]
        # Note: Scoreboard doesn't provide player-specific goals, so final_player_goals remains empty
        print(f"   Scoreboard confidence: {scoreboard_score['confidence']:.2f}")
        print(f"   Detection rate: {scoreboard_score.get('detection_rate', 0):.2f}")

    # 3. Enhanced goal detection (prioritize over regular detection even if 0 goals)
    elif enhanced_goal_stats is not None:
        # Enhanced detection is more accurate, use it even if it detects 0 goals
        if enhanced_goal_stats.get("final_total_goals", 0) >= 0:
            print("📊 Using enhanced goal detection final counts as final goal count")
            final_team_goals = enhanced_goal_stats["final_team_goals"].copy()
            final_player_goals = enhanced_goal_stats["final_player_goals"].copy()
        else:
            print(
                "📊 Using enhanced goal detection real-time counts as final goal count"
            )
            final_team_goals = enhanced_goal_stats["team_goals"].copy()
            final_player_goals = enhanced_goal_stats["player_goals"].copy()

    # 4. Regular goal detection (lowest priority - only if enhanced detection unavailable)
    else:
        print(
            "📊 Using regular goal detection as final goal count (enhanced detection unavailable)"
        )
        final_team_goals = pass_counter.team_goals.copy()
        final_player_goals = pass_counter.player_goals.copy()

    # Print summary
    total_final_goals = sum(final_team_goals.values())
    print(f"\n🏆 Final Goal Summary:")
    print(f"   Team 1: {final_team_goals.get(1, 0)} goals")
    print(f"   Team 2: {final_team_goals.get(2, 0)} goals")
    print(f"   Total: {total_final_goals} goals")

    if final_player_goals:
        print(f"   Player goals:")
        for player_id, data in final_player_goals.items():
            print(
                f"     Player {player_id} (Team {data.get('team', '?')}): {data.get('goals', 0)} goals"
            )

    return final_team_goals, final_player_goals


def export_consolidated_goal_statistics(
    video_name,
    pass_counter,
    enhanced_goal_stats,
    final_team_goals,
    final_player_goals,
    tackle_counter,
    scoreboard_analyzer=None,
    output_dir="data/output",
    uploader=None,
    bucket_name=None,
    upload_folder_prefix="football_analysis",
):
    """
    Export consolidated goal statistics to exactly two CSV files as requested.
    Optionally upload to DigitalOcean Spaces.

    Args:
        video_name (str): Name of the video being processed
        pass_counter: PassCounter instance
        enhanced_goal_stats: Enhanced goal statistics from GoalDetector
        final_team_goals (dict): Final team goal counts
        final_player_goals (dict): Final player goal counts
        tackle_counter: TackleCounter instance
        scoreboard_analyzer: ScoreboardAnalyzer instance (optional)
        output_dir (str): Output directory for CSV files
        uploader: DigitalOceanSpacesUploader instance (optional)
        bucket_name (str): Bucket name for upload (optional)
        upload_folder_prefix (str): Folder prefix for uploads

    Returns:
        tuple: (team_csv_path, player_csv_path)
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Get scoreboard information
    scoreboard_score = None
    scoreboard_stats = {}
    if scoreboard_analyzer and scoreboard_analyzer.is_scoreboard_available():
        scoreboard_score = scoreboard_analyzer.get_final_score()
        scoreboard_stats = scoreboard_analyzer.get_statistics()

    # Consolidated team statistics CSV
    team_csv_path = os.path.join(output_dir, f"{video_name}_team_stats.csv")

    with open(team_csv_path, "w", newline="") as csvfile:
        fieldnames = [
            "team",
            "passes",
            "goals_regular",
            "goals_enhanced",
            "goals_scoreboard",
            "goals_final",
            "tackles",
            "interceptions",
            "goal_events_count",
            "avg_goal_confidence",
            "scoreboard_detected",
            "scoreboard_confidence",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for team_id in [1, 2]:
            # Calculate team-specific goal events and confidence
            team_goal_events = [
                event
                for event in enhanced_goal_stats.get("goal_events", [])
                if event.get("team") == team_id
            ]
            avg_confidence = sum(
                event.get("confidence_score", 0) for event in team_goal_events
            ) / max(1, len(team_goal_events))

            # Get scoreboard goals for this team
            scoreboard_goals = 0
            scoreboard_confidence = 0
            if scoreboard_score:
                if team_id == 1:
                    scoreboard_goals = scoreboard_score.get("team1_score", 0)
                else:
                    scoreboard_goals = scoreboard_score.get("team2_score", 0)
                scoreboard_confidence = scoreboard_score.get("confidence", 0)

            writer.writerow(
                {
                    "team": team_id,
                    "passes": pass_counter.team_passes.get(team_id, 0),
                    "goals_regular": pass_counter.team_goals.get(team_id, 0),
                    "goals_enhanced": enhanced_goal_stats.get("team_goals", {}).get(
                        team_id, 0
                    ),
                    "goals_scoreboard": scoreboard_goals,
                    "goals_final": final_team_goals.get(team_id, 0),
                    "tackles": tackle_counter.team_tackles.get(team_id, 0),
                    "interceptions": tackle_counter.team_interceptions.get(team_id, 0),
                    "goal_events_count": len(team_goal_events),
                    "avg_goal_confidence": round(avg_confidence, 3),
                    "scoreboard_detected": bool(scoreboard_score),
                    "scoreboard_confidence": round(scoreboard_confidence, 3),
                }
            )

    # Consolidated player statistics CSV
    player_csv_path = os.path.join(output_dir, f"{video_name}_player_stats.csv")

    # Collect all player IDs from various sources
    all_player_ids = set()
    all_player_ids.update(pass_counter.player_passes.keys())
    all_player_ids.update(pass_counter.player_goals.keys())
    all_player_ids.update(enhanced_goal_stats.get("player_goals", {}).keys())
    all_player_ids.update(final_player_goals.keys())
    all_player_ids.update(tackle_counter.player_tackles.keys())
    all_player_ids.update(tackle_counter.player_interceptions.keys())

    with open(player_csv_path, "w", newline="") as csvfile:
        fieldnames = [
            "player_id",
            "team",
            "passes",
            "goals_regular",
            "goals_enhanced",
            "goals_final",
            "tackles",
            "interceptions",
            "goal_events_count",
            "avg_goal_confidence",
            "scoreboard_detected",
            "scoreboard_confidence",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for player_id in sorted(all_player_ids):
            # Determine player team
            team = (
                pass_counter.player_passes.get(player_id, {}).get("team")
                or pass_counter.player_goals.get(player_id, {}).get("team")
                or enhanced_goal_stats.get("player_goals", {})
                .get(player_id, {})
                .get("team")
                or final_player_goals.get(player_id, {}).get("team")
                or tackle_counter.player_tackles.get(player_id, {}).get("team")
                or tackle_counter.player_interceptions.get(player_id, {}).get("team")
                or "Unknown"
            )

            # Calculate player-specific goal events and confidence
            player_goal_events = [
                event
                for event in enhanced_goal_stats.get("goal_events", [])
                if event.get("player_id") == player_id
            ]
            avg_confidence = sum(
                event.get("confidence_score", 0) for event in player_goal_events
            ) / max(1, len(player_goal_events))

            writer.writerow(
                {
                    "player_id": player_id,
                    "team": team,
                    "passes": pass_counter.player_passes.get(player_id, {}).get(
                        "passes", 0
                    ),
                    "goals_regular": pass_counter.player_goals.get(player_id, {}).get(
                        "goals", 0
                    ),
                    "goals_enhanced": enhanced_goal_stats.get("player_goals", {})
                    .get(player_id, {})
                    .get("goals", 0),
                    "goals_final": final_player_goals.get(player_id, {}).get(
                        "goals", 0
                    ),
                    "tackles": tackle_counter.player_tackles.get(player_id, {}).get(
                        "tackles", 0
                    ),
                    "interceptions": tackle_counter.player_interceptions.get(
                        player_id, {}
                    ).get("interceptions", 0),
                    "goal_events_count": len(player_goal_events),
                    "avg_goal_confidence": round(avg_confidence, 3),
                    "scoreboard_detected": bool(scoreboard_score),
                    "scoreboard_confidence": (
                        round(scoreboard_score.get("confidence", 0), 3)
                        if scoreboard_score
                        else 0
                    ),
                }
            )

    print(f"📊 Consolidated statistics exported:")
    print(f"   Team stats: {team_csv_path}")
    print(f"   Player stats: {player_csv_path}")

    # Upload to DigitalOcean Spaces if uploader is provided
    if uploader is not None:
        print(f"\n☁️  Uploading CSV files to DigitalOcean Spaces...")
        upload_results = uploader.upload_csv_files(
            team_csv_path=team_csv_path,
            player_csv_path=player_csv_path,
            video_name=video_name,
            bucket_name=bucket_name,
            folder_prefix=upload_folder_prefix,
        )

        if upload_results.get("team_csv", False):
            print(f"✅ Team CSV uploaded successfully")
        else:
            print(f"❌ Failed to upload team CSV")

        if upload_results.get("player_csv", False):
            print(f"✅ Player CSV uploaded successfully")
        else:
            print(f"❌ Failed to upload player CSV")

    return team_csv_path, player_csv_path
