import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .unified_statistics_manager import AnalysisResults


class EnhancedCSVExporter:
    """
    Enhanced CSV exporter that provides comprehensive export functionality
    for all enhanced football statistics with quality metrics and metadata.
    """

    def __init__(self, output_dir: str = "data/output"):
        """
        Initialize the Enhanced CSV Exporter.

        Args:
            output_dir (str): Base output directory for CSV files
        """
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def export_all_statistics(
        self, analysis_results: AnalysisResults, include_metadata: bool = True
    ) -> Dict[str, str]:
        """
        Export all statistics from analysis results to CSV files.

        Args:
            analysis_results (AnalysisResults): Complete analysis results
            include_metadata (bool): Whether to include metadata files

        Returns:
            Dict[str, str]: Dictionary mapping export type to file path
        """
        os.makedirs(self.output_dir, exist_ok=True)
        exported_files = {}

        print("📊 Exporting enhanced statistics to CSV...")

        # Export team statistics
        team_stats_path = self._export_team_statistics(analysis_results)
        exported_files["team_statistics"] = team_stats_path

        # Export player statistics
        player_stats_path = self._export_player_statistics(analysis_results)
        exported_files["player_statistics"] = player_stats_path

        # Export event data
        events_paths = self._export_event_data(analysis_results)
        exported_files.update(events_paths)

        # Export quality metrics
        quality_path = self._export_quality_metrics(analysis_results)
        exported_files["quality_metrics"] = quality_path

        # Export match summary
        summary_path = self._export_match_summary(analysis_results)
        exported_files["match_summary"] = summary_path

        # Export frame-by-frame data
        frame_data_path = self._export_frame_by_frame_data(analysis_results)
        exported_files["frame_data"] = frame_data_path

        if include_metadata:
            # Export metadata and configuration
            metadata_paths = self._export_metadata(analysis_results)
            exported_files.update(metadata_paths)

        print(f"✅ Enhanced statistics exported to {len(exported_files)} files")
        return exported_files

    def _export_team_statistics(self, analysis_results: AnalysisResults) -> str:
        """Export comprehensive team statistics."""
        output_path = os.path.join(
            self.output_dir, f"enhanced_team_stats_{self.timestamp}.csv"
        )

        team_data = []
        for team_id, stats in analysis_results.team_statistics.items():
            if hasattr(stats, "__dict__"):
                # Convert dataclass to dict
                team_dict = stats.__dict__.copy()
            else:
                team_dict = stats

            # Flatten nested dictionaries and add calculated metrics
            flattened_stats = self._flatten_team_stats(team_dict)
            team_data.append(flattened_stats)

        if team_data:
            df = pd.DataFrame(team_data)
            df.to_csv(output_path, index=False)
            print(f"📈 Team statistics exported: {output_path}")

        return output_path

    def _export_player_statistics(self, analysis_results: AnalysisResults) -> str:
        """Export comprehensive player statistics."""
        output_path = os.path.join(
            self.output_dir, f"enhanced_player_stats_{self.timestamp}.csv"
        )

        player_data = []
        for player_id, stats in analysis_results.player_statistics.items():
            if hasattr(stats, "__dict__"):
                # Convert dataclass to dict
                player_dict = stats.__dict__.copy()
            else:
                player_dict = stats

            # Add calculated performance metrics
            enhanced_stats = self._enhance_player_stats(player_dict)
            player_data.append(enhanced_stats)

        if player_data:
            df = pd.DataFrame(player_data)
            # Sort by team and then by performance metrics
            if "team_id" in df.columns:
                df = df.sort_values(["team_id", "player_id"])
            df.to_csv(output_path, index=False)
            print(f"👤 Player statistics exported: {output_path}")

        return output_path

    def _export_event_data(self, analysis_results: AnalysisResults) -> Dict[str, str]:
        """Export detailed event data."""
        exported_files = {}

        # Export pass events
        if analysis_results.pass_events:
            pass_path = os.path.join(
                self.output_dir, f"pass_events_{self.timestamp}.csv"
            )
            self._export_pass_events(analysis_results.pass_events, pass_path)
            exported_files["pass_events"] = pass_path

        # Export tackle events
        if analysis_results.tackle_events:
            tackle_path = os.path.join(
                self.output_dir, f"tackle_events_{self.timestamp}.csv"
            )
            self._export_tackle_events(analysis_results.tackle_events, tackle_path)
            exported_files["tackle_events"] = tackle_path

        # Export interception events
        if analysis_results.interception_events:
            interception_path = os.path.join(
                self.output_dir, f"interception_events_{self.timestamp}.csv"
            )
            self._export_interception_events(
                analysis_results.interception_events, interception_path
            )
            exported_files["interception_events"] = interception_path

        # Export possession events
        if analysis_results.possession_events:
            possession_path = os.path.join(
                self.output_dir, f"possession_events_{self.timestamp}.csv"
            )
            self._export_possession_events(
                analysis_results.possession_events, possession_path
            )
            exported_files["possession_events"] = possession_path

        return exported_files

    def _export_pass_events(self, pass_events: List, output_path: str):
        """Export pass events with enhanced data."""
        with open(output_path, "w", newline="") as csvfile:
            fieldnames = [
                "frame_num",
                "passer_id",
                "receiver_id",
                "passer_team",
                "receiver_team",
                "pass_distance",
                "pass_velocity",
                "pass_accuracy",
                "confidence_score",
                "pass_type",
                "trajectory_validated",
                "timestamp_seconds",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for event in pass_events:
                if isinstance(event, dict):
                    event_data = event
                else:
                    event_data = event.__dict__ if hasattr(event, "__dict__") else {}

                # Calculate timestamp in seconds (assuming 24 fps)
                timestamp_seconds = event_data.get("frame_num", 0) / 24.0

                writer.writerow(
                    {
                        "frame_num": event_data.get("frame_num", 0),
                        "passer_id": event_data.get("passer_id", -1),
                        "receiver_id": event_data.get("receiver_id", -1),
                        "passer_team": event_data.get("passer_team", 0),
                        "receiver_team": event_data.get("receiver_team", 0),
                        "pass_distance": round(event_data.get("pass_distance", 0), 2),
                        "pass_velocity": round(event_data.get("pass_velocity", 0), 2),
                        "pass_accuracy": round(event_data.get("pass_accuracy", 0), 3),
                        "confidence_score": round(
                            event_data.get("confidence_score", 0), 3
                        ),
                        "pass_type": event_data.get("pass_type", "unknown"),
                        "trajectory_validated": bool(
                            event_data.get("trajectory_segment")
                        ),
                        "timestamp_seconds": round(timestamp_seconds, 2),
                    }
                )

        print(f"⚽ Pass events exported: {output_path}")

    def _export_tackle_events(self, tackle_events: List, output_path: str):
        """Export tackle events with enhanced data."""
        with open(output_path, "w", newline="") as csvfile:
            fieldnames = [
                "frame_num",
                "tackler_id",
                "tackled_player_id",
                "tackler_team",
                "tackled_team",
                "tackle_distance",
                "ball_velocity_before",
                "ball_velocity_after",
                "success_probability",
                "confidence_score",
                "tackle_type",
                "timestamp_seconds",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for event in tackle_events:
                if isinstance(event, dict):
                    event_data = event
                else:
                    event_data = event.__dict__ if hasattr(event, "__dict__") else {}

                timestamp_seconds = event_data.get("frame_num", 0) / 24.0

                writer.writerow(
                    {
                        "frame_num": event_data.get("frame_num", 0),
                        "tackler_id": event_data.get("tackler_id", -1),
                        "tackled_player_id": event_data.get("tackled_player_id", -1),
                        "tackler_team": event_data.get("tackler_team", 0),
                        "tackled_team": event_data.get("tackled_team", 0),
                        "tackle_distance": round(
                            event_data.get("tackle_distance", 0), 2
                        ),
                        "ball_velocity_before": round(
                            event_data.get("ball_velocity_before", 0), 2
                        ),
                        "ball_velocity_after": round(
                            event_data.get("ball_velocity_after", 0), 2
                        ),
                        "success_probability": round(
                            event_data.get("success_probability", 0), 3
                        ),
                        "confidence_score": round(
                            event_data.get("confidence_score", 0), 3
                        ),
                        "tackle_type": event_data.get("tackle_type", "unknown"),
                        "timestamp_seconds": round(timestamp_seconds, 2),
                    }
                )

        print(f"🛡️ Tackle events exported: {output_path}")

    def _export_interception_events(self, interception_events: List, output_path: str):
        """Export interception events with enhanced data."""
        with open(output_path, "w", newline="") as csvfile:
            fieldnames = [
                "frame_num",
                "interceptor_id",
                "interceptor_team",
                "original_target_team",
                "interception_distance",
                "ball_trajectory_change",
                "confidence_score",
                "interception_type",
                "timestamp_seconds",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for event in interception_events:
                if isinstance(event, dict):
                    event_data = event
                else:
                    event_data = event.__dict__ if hasattr(event, "__dict__") else {}

                timestamp_seconds = event_data.get("frame_num", 0) / 24.0

                writer.writerow(
                    {
                        "frame_num": event_data.get("frame_num", 0),
                        "interceptor_id": event_data.get("interceptor_id", -1),
                        "interceptor_team": event_data.get("interceptor_team", 0),
                        "original_target_team": event_data.get(
                            "original_target_team", 0
                        ),
                        "interception_distance": round(
                            event_data.get("interception_distance", 0), 2
                        ),
                        "ball_trajectory_change": round(
                            event_data.get("ball_trajectory_change", 0), 2
                        ),
                        "confidence_score": round(
                            event_data.get("confidence_score", 0), 3
                        ),
                        "interception_type": event_data.get(
                            "interception_type", "unknown"
                        ),
                        "timestamp_seconds": round(timestamp_seconds, 2),
                    }
                )

        print(f"🔄 Interception events exported: {output_path}")

    def _export_possession_events(self, possession_events: List, output_path: str):
        """Export possession events with enhanced data."""
        with open(output_path, "w", newline="") as csvfile:
            fieldnames = [
                "start_frame",
                "end_frame",
                "player_id",
                "team_id",
                "duration_frames",
                "duration_seconds",
                "ball_touches",
                "possession_quality",
                "start_timestamp",
                "end_timestamp",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for event in possession_events:
                if isinstance(event, dict):
                    event_data = event
                else:
                    event_data = event.__dict__ if hasattr(event, "__dict__") else {}

                start_frame = event_data.get("start_frame", 0)
                end_frame = event_data.get("end_frame", 0)
                duration_frames = event_data.get(
                    "duration_frames", end_frame - start_frame
                )

                writer.writerow(
                    {
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "player_id": event_data.get("player_id", -1),
                        "team_id": event_data.get("team_id", 0),
                        "duration_frames": duration_frames,
                        "duration_seconds": round(duration_frames / 24.0, 2),
                        "ball_touches": event_data.get("ball_touches", 1),
                        "possession_quality": round(
                            event_data.get("possession_quality", 0), 3
                        ),
                        "start_timestamp": round(start_frame / 24.0, 2),
                        "end_timestamp": round(end_frame / 24.0, 2),
                    }
                )

        print(f"🏃 Possession events exported: {output_path}")

    def _export_quality_metrics(self, analysis_results: AnalysisResults) -> str:
        """Export analysis quality metrics."""
        output_path = os.path.join(
            self.output_dir, f"quality_metrics_{self.timestamp}.csv"
        )

        quality_data = []

        # Ball tracking quality
        ball_quality = analysis_results.ball_tracking_quality
        if ball_quality:
            quality_data.append(
                {
                    "metric_category": "ball_tracking",
                    "metric_name": "high_confidence_ratio",
                    "metric_value": ball_quality.get("high_confidence_ratio", 0),
                    "description": "Ratio of high-confidence ball detections",
                }
            )
            quality_data.append(
                {
                    "metric_category": "ball_tracking",
                    "metric_name": "interpolated_ratio",
                    "metric_value": ball_quality.get("interpolated_ratio", 0),
                    "description": "Ratio of interpolated ball positions",
                }
            )
            quality_data.append(
                {
                    "metric_category": "ball_tracking",
                    "metric_name": "tracking_continuity",
                    "metric_value": ball_quality.get("tracking_continuity", 0),
                    "description": "Ball tracking continuity score",
                }
            )

        # Analysis quality metrics
        analysis_quality = analysis_results.analysis_quality_metrics
        if analysis_quality:
            for metric_name, metric_value in analysis_quality.items():
                if isinstance(metric_value, (int, float)):
                    quality_data.append(
                        {
                            "metric_category": "analysis_quality",
                            "metric_name": metric_name,
                            "metric_value": metric_value,
                            "description": f"Analysis quality metric: {metric_name}",
                        }
                    )

        # Frame coverage metrics
        frame_coverage = analysis_results.frame_coverage
        if frame_coverage:
            for metric_name, metric_value in frame_coverage.items():
                if isinstance(metric_value, (int, float)):
                    quality_data.append(
                        {
                            "metric_category": "frame_coverage",
                            "metric_name": metric_name,
                            "metric_value": metric_value,
                            "description": f"Frame coverage metric: {metric_name}",
                        }
                    )

        if quality_data:
            df = pd.DataFrame(quality_data)
            df.to_csv(output_path, index=False)
            print(f"📊 Quality metrics exported: {output_path}")

        return output_path

    def _export_match_summary(self, analysis_results: AnalysisResults) -> str:
        """Export match summary with key statistics."""
        output_path = os.path.join(
            self.output_dir, f"match_summary_{self.timestamp}.csv"
        )

        summary_data = []

        # Overall match statistics
        summary_data.append(
            {
                "category": "match_overview",
                "metric": "total_pass_events",
                "value": len(analysis_results.pass_events),
                "description": "Total number of pass events detected",
            }
        )

        summary_data.append(
            {
                "category": "match_overview",
                "metric": "total_tackle_events",
                "value": len(analysis_results.tackle_events),
                "description": "Total number of tackle events detected",
            }
        )

        summary_data.append(
            {
                "category": "match_overview",
                "metric": "total_interception_events",
                "value": len(analysis_results.interception_events),
                "description": "Total number of interception events detected",
            }
        )

        # Team-specific summaries
        for team_id, team_stats in analysis_results.team_statistics.items():
            if hasattr(team_stats, "__dict__"):
                stats_dict = team_stats.__dict__
            else:
                stats_dict = team_stats

            summary_data.append(
                {
                    "category": f"team_{team_id}",
                    "metric": "possession_percentage",
                    "value": stats_dict.get("possession_percentage", 0),
                    "description": f"Team {team_id} possession percentage",
                }
            )

            summary_data.append(
                {
                    "category": f"team_{team_id}",
                    "metric": "pass_accuracy",
                    "value": stats_dict.get("pass_accuracy", 0),
                    "description": f"Team {team_id} pass accuracy",
                }
            )

        # Quality summary
        quality_metrics = analysis_results.analysis_quality_metrics
        if quality_metrics:
            summary_data.append(
                {
                    "category": "quality",
                    "metric": "overall_quality_score",
                    "value": quality_metrics.get("analysis_completeness", 0),
                    "description": "Overall analysis quality score",
                }
            )

        if summary_data:
            df = pd.DataFrame(summary_data)
            df.to_csv(output_path, index=False)
            print(f"📋 Match summary exported: {output_path}")

        return output_path

    def _export_frame_by_frame_data(self, analysis_results: AnalysisResults) -> str:
        """Export frame-by-frame analysis data."""
        output_path = os.path.join(self.output_dir, f"frame_data_{self.timestamp}.csv")

        # This would contain frame-by-frame ball tracking quality, possession changes, etc.
        # For now, we'll create a simplified version
        frame_data = []

        # Combine all events by frame for frame-by-frame analysis
        all_events = {}

        # Add pass events
        for event in analysis_results.pass_events:
            frame_num = (
                event.get("frame_num")
                if isinstance(event, dict)
                else getattr(event, "frame_num", 0)
            )
            if frame_num not in all_events:
                all_events[frame_num] = {"frame_num": frame_num, "events": []}
            all_events[frame_num]["events"].append("pass")

        # Add tackle events
        for event in analysis_results.tackle_events:
            frame_num = (
                event.get("frame_num")
                if isinstance(event, dict)
                else getattr(event, "frame_num", 0)
            )
            if frame_num not in all_events:
                all_events[frame_num] = {"frame_num": frame_num, "events": []}
            all_events[frame_num]["events"].append("tackle")

        # Add interception events
        for event in analysis_results.interception_events:
            frame_num = (
                event.get("frame_num")
                if isinstance(event, dict)
                else getattr(event, "frame_num", 0)
            )
            if frame_num not in all_events:
                all_events[frame_num] = {"frame_num": frame_num, "events": []}
            all_events[frame_num]["events"].append("interception")

        # Create frame data
        for frame_num, frame_info in all_events.items():
            frame_data.append(
                {
                    "frame_num": frame_num,
                    "timestamp_seconds": round(frame_num / 24.0, 2),
                    "event_count": len(frame_info["events"]),
                    "event_types": ",".join(frame_info["events"]),
                    "has_pass": "pass" in frame_info["events"],
                    "has_tackle": "tackle" in frame_info["events"],
                    "has_interception": "interception" in frame_info["events"],
                }
            )

        if frame_data:
            df = pd.DataFrame(frame_data)
            df = df.sort_values("frame_num")
            df.to_csv(output_path, index=False)
            print(f"🎬 Frame-by-frame data exported: {output_path}")

        return output_path

    def _export_metadata(self, analysis_results: AnalysisResults) -> Dict[str, str]:
        """Export metadata and configuration files."""
        exported_files = {}

        # Export configuration
        config_path = os.path.join(
            self.output_dir, f"analysis_config_{self.timestamp}.json"
        )
        if hasattr(analysis_results.configuration, "__dict__"):
            config_dict = analysis_results.configuration.__dict__
        else:
            config_dict = analysis_results.configuration

        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        exported_files["configuration"] = config_path

        # Export analysis metadata
        metadata_path = os.path.join(
            self.output_dir, f"analysis_metadata_{self.timestamp}.json"
        )
        metadata = {
            "analysis_timestamp": analysis_results.timestamp,
            "total_events_detected": (
                len(analysis_results.pass_events)
                + len(analysis_results.tackle_events)
                + len(analysis_results.interception_events)
            ),
            "frame_coverage": analysis_results.frame_coverage,
            "ball_tracking_quality": analysis_results.ball_tracking_quality,
            "analysis_quality_metrics": analysis_results.analysis_quality_metrics,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        exported_files["metadata"] = metadata_path

        print(f"📄 Metadata exported: {len(exported_files)} files")
        return exported_files

    def _flatten_team_stats(self, team_dict: Dict) -> Dict:
        """Flatten team statistics dictionary."""
        flattened = {}

        for key, value in team_dict.items():
            if isinstance(value, dict):
                # Flatten nested dictionaries
                for nested_key, nested_value in value.items():
                    flattened[f"{key}_{nested_key}"] = nested_value
            else:
                flattened[key] = value

        return flattened

    def _enhance_player_stats(self, player_dict: Dict) -> Dict:
        """Enhance player statistics with calculated metrics."""
        enhanced = player_dict.copy()

        # Calculate additional performance metrics
        passes_attempted = enhanced.get("passes_attempted", 0)
        passes_completed = enhanced.get("passes_completed", 0)

        if passes_attempted > 0:
            enhanced["pass_completion_rate"] = passes_completed / passes_attempted
        else:
            enhanced["pass_completion_rate"] = 0.0

        # Calculate defensive efficiency
        tackles_attempted = enhanced.get("tackles_attempted", 0)
        tackles_successful = enhanced.get("tackles_successful", 0)
        interceptions = enhanced.get("interceptions", 0)

        total_defensive_actions = tackles_attempted + interceptions
        successful_defensive_actions = tackles_successful + interceptions

        if total_defensive_actions > 0:
            enhanced["defensive_efficiency"] = (
                successful_defensive_actions / total_defensive_actions
            )
        else:
            enhanced["defensive_efficiency"] = 0.0

        # Calculate overall performance score (simplified)
        pass_score = enhanced.get("pass_accuracy", 0) * 0.4
        defensive_score = enhanced.get("defensive_efficiency", 0) * 0.3
        possession_score = min(enhanced.get("ball_control_quality", 0), 1.0) * 0.3

        enhanced["overall_performance_score"] = (
            pass_score + defensive_score + possession_score
        )

        return enhanced
