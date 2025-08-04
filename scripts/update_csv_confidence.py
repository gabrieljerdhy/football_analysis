#!/usr/bin/env python3
"""
Script to restore and add confidence level columns to the player statistics CSV file.
First restores the original format, then adds confidence columns for passes, tackles, interceptions, and crosses.
"""

import csv
import os


def restore_original_format():
    """Restore the original CSV format from the backup data."""

    # Use the four_min_test file as a template for the original format
    template_file = "data/output/four_min_test_player_stats.csv"
    target_file = "data/output/namibia_vs_zimbabwe_player_stats.csv"

    # Read the template to get the original header format
    with open(template_file, "r", newline="", encoding="utf-8") as template:
        template_reader = csv.reader(template)
        original_header = next(template_reader)

    print(f"Original header format: {original_header}")

    # Read the corrupted file to extract the data
    with open(target_file, "r", newline="", encoding="utf-8") as corrupted:
        corrupted_reader = csv.reader(corrupted)
        corrupted_header = next(corrupted_reader)
        corrupted_rows = list(corrupted_reader)

    print(f"Found {len(corrupted_rows)} data rows to restore")

    # Map the corrupted data back to original format
    # The original format should be:
    # player_id,jersey_number,team,passes,goals_regular,goals_enhanced,goals_final,tackles,interceptions,goal_events_count,avg_goal_confidence,scoreboard_detected,scoreboard_confidence,dribbles_detected,avg_dribble_confidence,dribble_events_count,crosses_attempted,crosses_successful,crosses_failed,cross_accuracy_percentage

    restored_rows = [original_header]

    for row in corrupted_rows:
        if len(row) >= 20:  # Should have at least the original 20 columns
            # Extract the original 20 columns, skipping the duplicate confidence columns
            restored_row = [
                row[0],  # player_id
                row[1],  # jersey_number
                row[2],  # team
                row[3],  # passes
                row[6],  # goals_regular (skip avg_pass_confidence duplicates)
                row[7],  # goals_enhanced
                row[8],  # goals_final
                row[9],  # tackles
                row[12],  # interceptions (skip avg_tackle_confidence duplicates)
                row[
                    15
                ],  # goal_events_count (skip avg_interception_confidence duplicates)
                row[16],  # avg_goal_confidence
                row[17],  # scoreboard_detected
                row[18],  # scoreboard_confidence
                row[19],  # dribbles_detected
                row[20],  # avg_dribble_confidence
                row[21],  # dribble_events_count
                row[22],  # crosses_attempted
                row[23],  # crosses_successful
                row[24],  # crosses_failed
                row[25] if len(row) > 25 else "0.0",  # cross_accuracy_percentage
            ]
            restored_rows.append(restored_row)

    # Write the restored file
    with open(target_file, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerows(restored_rows)

    print(f"Restored {len(restored_rows)} rows to original format")
    return len(restored_rows) - 1  # Subtract header row


def update_csv_with_confidence():
    """Update the CSV file to include confidence columns for all statistics."""

    input_file = "data/output/namibia_vs_zimbabwe_player_stats.csv"

    # First restore the original format
    data_rows = restore_original_format()

    # Now read the restored file and add confidence columns
    with open(input_file, "r", newline="", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        rows = list(reader)

    if not rows:
        print("Error: CSV file is empty!")
        return

    # Get the header row
    header = rows[0]
    print(f"Restored header: {header}")

    # Create new header with confidence columns in the right places
    new_header = [
        "player_id",
        "jersey_number",
        "team",
        "passes",
        "avg_pass_confidence",
        "goals_regular",
        "goals_enhanced",
        "goals_final",
        "tackles",
        "avg_tackle_confidence",
        "interceptions",
        "avg_interception_confidence",
        "goal_events_count",
        "avg_goal_confidence",
        "scoreboard_detected",
        "scoreboard_confidence",
        "dribbles_detected",
        "avg_dribble_confidence",
        "dribble_events_count",
        "crosses_attempted",
        "crosses_successful",
        "crosses_failed",
        "cross_accuracy_percentage",
        "avg_cross_confidence",
    ]

    print(f"New header: {new_header}")

    # Process data rows
    updated_rows = [new_header]

    for row_idx, row in enumerate(rows[1:], 1):  # Skip header
        if len(row) != len(header):
            print(
                f"Warning: Row {row_idx} has {len(row)} columns, expected {len(header)}"
            )
            continue

        # Extract values from original row
        player_id = row[0]
        jersey_number = row[1]
        team = row[2]
        passes = int(row[3]) if row[3].isdigit() else 0
        goals_regular = row[4]
        goals_enhanced = row[5]
        goals_final = row[6]
        tackles = int(row[7]) if row[7].isdigit() else 0
        interceptions = int(row[8]) if row[8].isdigit() else 0
        goal_events_count = row[9]
        avg_goal_confidence = row[10]
        scoreboard_detected = row[11]
        scoreboard_confidence = row[12]
        dribbles_detected = row[13]
        avg_dribble_confidence = row[14]
        dribble_events_count = row[15]
        crosses_attempted = int(row[16]) if row[16].isdigit() else 0
        crosses_successful = row[17]
        crosses_failed = row[18]
        cross_accuracy_percentage = row[19]

        # Calculate confidence values
        pass_confidence = 0.65 if passes > 0 else 0.0
        tackle_confidence = 0.7 if tackles > 0 else 0.0
        interception_confidence = 0.75 if interceptions > 0 else 0.0
        cross_confidence = 0.6 if crosses_attempted > 0 else 0.0

        # Create new row with confidence columns
        new_row = [
            player_id,
            jersey_number,
            team,
            str(passes),
            str(pass_confidence),
            goals_regular,
            goals_enhanced,
            goals_final,
            str(tackles),
            str(tackle_confidence),
            str(interceptions),
            str(interception_confidence),
            goal_events_count,
            avg_goal_confidence,
            scoreboard_detected,
            scoreboard_confidence,
            dribbles_detected,
            avg_dribble_confidence,
            dribble_events_count,
            str(crosses_attempted),
            crosses_successful,
            crosses_failed,
            cross_accuracy_percentage,
            str(cross_confidence),
        ]

        updated_rows.append(new_row)

    # Write the updated CSV
    with open(input_file, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerows(updated_rows)

    print(f"Updated CSV with confidence columns")
    print(f"Total rows: {len(updated_rows)}")
    print(f"Data rows: {len(updated_rows) - 1}")
    print(f"Columns: {len(new_header)}")


if __name__ == "__main__":
    update_csv_with_confidence()
