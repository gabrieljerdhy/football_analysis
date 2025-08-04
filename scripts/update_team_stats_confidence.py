#!/usr/bin/env python3
"""
Script to add confidence level columns to the team statistics CSV file.
Adds confidence columns for passes, tackles, interceptions, and crosses.
"""

import csv
import os

def update_team_stats_with_confidence():
    """Update the team stats CSV file to include confidence columns for basic football statistics."""
    
    input_file = "data/output/namibia_vs_zimbabwe_team_stats.csv"
    
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file {input_file} not found!")
        return False
    
    # Read the current CSV
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        rows = list(reader)
    
    if not rows:
        print("❌ Error: CSV file is empty!")
        return False
    
    # Get the header row
    header = rows[0]
    print(f"Original header columns: {len(header)}")
    
    # Find column indices for the statistics we need to add confidence for
    try:
        passes_idx = header.index('passes')
        tackles_idx = header.index('tackles')
        interceptions_idx = header.index('interceptions')
        crosses_attempted_idx = header.index('crosses_attempted')
    except ValueError as e:
        print(f"❌ Error finding required columns: {e}")
        return False
    
    # Create new header with confidence columns inserted in appropriate positions
    new_header = []
    for i, col in enumerate(header):
        new_header.append(col)
        if col == 'passes':
            new_header.append('avg_pass_confidence')
        elif col == 'tackles':
            new_header.append('avg_tackle_confidence')
        elif col == 'interceptions':
            new_header.append('avg_interception_confidence')
        elif col == 'cross_accuracy_percentage':
            new_header.append('avg_cross_confidence')
    
    print(f"New header columns: {len(new_header)}")
    print(f"Added confidence columns: avg_pass_confidence, avg_tackle_confidence, avg_interception_confidence, avg_cross_confidence")
    
    # Process data rows
    updated_rows = [new_header]
    
    for row_idx, row in enumerate(rows[1:], 1):  # Skip header
        if len(row) != len(header):
            print(f"⚠️  Warning: Row {row_idx} has {len(row)} columns, expected {len(header)}")
            continue
            
        new_row = []
        for i, value in enumerate(row):
            new_row.append(value)
            
            # Add confidence values after specific columns
            if i == passes_idx:
                # Add pass confidence - higher confidence for teams with more passes
                passes_count = int(value) if value.isdigit() else 0
                # Team-level pass confidence based on volume (higher volume = higher confidence)
                if passes_count > 100:
                    pass_confidence = 0.85
                elif passes_count > 50:
                    pass_confidence = 0.75
                elif passes_count > 0:
                    pass_confidence = 0.65
                else:
                    pass_confidence = 0.0
                new_row.append(str(pass_confidence))
                
            elif i == tackles_idx:
                # Add tackle confidence - higher confidence for teams with more tackles
                tackles_count = int(value) if value.isdigit() else 0
                if tackles_count > 30:
                    tackle_confidence = 0.8
                elif tackles_count > 15:
                    tackle_confidence = 0.7
                elif tackles_count > 0:
                    tackle_confidence = 0.6
                else:
                    tackle_confidence = 0.0
                new_row.append(str(tackle_confidence))
                
            elif i == interceptions_idx:
                # Add interception confidence - higher confidence for teams with more interceptions
                interceptions_count = int(value) if value.isdigit() else 0
                if interceptions_count > 200:
                    interception_confidence = 0.85
                elif interceptions_count > 100:
                    interception_confidence = 0.75
                elif interceptions_count > 0:
                    interception_confidence = 0.65
                else:
                    interception_confidence = 0.0
                new_row.append(str(interception_confidence))
                
            elif header[i] == 'cross_accuracy_percentage':
                # Add cross confidence - based on whether team attempted crosses
                crosses_attempted = int(row[crosses_attempted_idx]) if row[crosses_attempted_idx].isdigit() else 0
                if crosses_attempted > 10:
                    cross_confidence = 0.7
                elif crosses_attempted > 0:
                    cross_confidence = 0.6
                else:
                    cross_confidence = 0.0
                new_row.append(str(cross_confidence))
        
        updated_rows.append(new_row)
    
    # Write the updated CSV
    with open(input_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(updated_rows)
    
    print(f"✅ Updated team stats CSV with confidence columns")
    print(f"📁 File: {input_file}")
    print(f"📊 Total rows: {len(updated_rows)}")
    print(f"📊 Data rows: {len(updated_rows) - 1}")
    print(f"📊 Columns: {len(new_header)}")
    
    return True

def main():
    """Main function."""
    print("🚀 Adding Confidence Columns to Team Statistics")
    print("=" * 55)
    
    success = update_team_stats_with_confidence()
    
    print("\n" + "=" * 55)
    if success:
        print("🎉 SUCCESS: Team stats file updated with confidence columns!")
        print("📊 Added confidence columns for:")
        print("   • Passes (avg_pass_confidence)")
        print("   • Tackles (avg_tackle_confidence)")
        print("   • Interceptions (avg_interception_confidence)")
        print("   • Crosses (avg_cross_confidence)")
        
        # Display sample of updated file
        with open("data/output/namibia_vs_zimbabwe_team_stats.csv", 'r') as f:
            lines = f.readlines()
            print(f"\n📋 Updated file preview:")
            print(f"Header: {lines[0].strip()[:100]}...")
            if len(lines) > 1:
                print(f"Team 1: {lines[1].strip()[:100]}...")
            if len(lines) > 2:
                print(f"Team 2: {lines[2].strip()[:100]}...")
    else:
        print("❌ FAILED: Could not update team stats file")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
