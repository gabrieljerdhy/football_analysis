import os
import json
import pickle
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime

from .ball_movement_analyzer import EnhancedBallMovementAnalyzer
from .enhanced_pass_counter import EnhancedPassCounter
from .enhanced_tackle_counter import EnhancedTackleCounter
from .enhanced_statistics import EnhancedFootballStatistics
from src.player_ball_assigner import PlayerBallAssigner


@dataclass
class AnalysisConfiguration:
    """Configuration for the unified statistics analysis."""
    frame_rate: float = 24.0
    enable_enhanced_ball_tracking: bool = True
    enable_trajectory_analysis: bool = True
    enable_confidence_validation: bool = True
    
    # Analysis thresholds
    min_pass_confidence: float = 0.4
    min_tackle_confidence: float = 0.4
    min_possession_frames: int = 5
    
    # Export settings
    export_detailed_events: bool = True
    export_player_stats: bool = True
    export_team_stats: bool = True
    export_quality_metrics: bool = True


@dataclass
class AnalysisResults:
    """Complete analysis results structure."""
    timestamp: str
    configuration: AnalysisConfiguration
    
    # Core statistics
    team_statistics: Dict
    player_statistics: Dict
    
    # Event data
    pass_events: List
    tackle_events: List
    interception_events: List
    possession_events: List
    
    # Analysis metadata
    ball_tracking_quality: Dict
    analysis_quality_metrics: Dict
    frame_coverage: Dict
    
    # Summary metrics
    match_summary: Dict


class UnifiedStatisticsManager:
    """
    Unified statistics manager that coordinates all enhanced stat gathering modules
    and provides a centralized interface for football match analysis.
    """
    
    def __init__(self, config: Optional[AnalysisConfiguration] = None):
        """
        Initialize the Unified Statistics Manager.
        
        Args:
            config (AnalysisConfiguration): Configuration for analysis
        """
        self.config = config or AnalysisConfiguration()
        
        # Initialize component analyzers
        self.ball_analyzer = EnhancedBallMovementAnalyzer(
            frame_rate=self.config.frame_rate
        )
        self.pass_counter = EnhancedPassCounter(
            frame_rate=self.config.frame_rate
        )
        self.tackle_counter = EnhancedTackleCounter(
            frame_rate=self.config.frame_rate
        )
        self.comprehensive_stats = EnhancedFootballStatistics(
            frame_rate=self.config.frame_rate
        )
        
        # Player ball assignment
        self.player_assigner = PlayerBallAssigner()
        
        # Analysis state
        self.current_analysis: Optional[AnalysisResults] = None
        self.analysis_history: List[AnalysisResults] = []
        
        # Performance tracking
        self.performance_metrics = {
            "analysis_start_time": None,
            "analysis_end_time": None,
            "processing_time": 0.0,
            "frames_processed": 0,
            "events_detected": 0
        }
    
    def analyze_match(self, tracks: Dict, output_dir: str = "data/output", 
                     save_results: bool = True) -> AnalysisResults:
        """
        Perform comprehensive match analysis using all enhanced tracking modules.
        
        Args:
            tracks (Dict): Complete tracking data for the match
            output_dir (str): Directory to save analysis results
            save_results (bool): Whether to save results to files
            
        Returns:
            AnalysisResults: Complete analysis results
        """
        print("🚀 Starting unified match analysis...")
        self.performance_metrics["analysis_start_time"] = datetime.now()
        
        try:
            # Step 1: Prepare player assignments
            print("📋 Preparing player ball assignments...")
            player_assignments = self._generate_player_assignments(tracks)
            self.performance_metrics["frames_processed"] = len(player_assignments)
            
            # Step 2: Comprehensive analysis
            print("🔍 Performing comprehensive analysis...")
            comprehensive_results = self.comprehensive_stats.analyze_complete_match(
                tracks, player_assignments
            )
            
            # Step 3: Extract detailed event data
            print("📊 Extracting detailed event data...")
            detailed_events = self._extract_detailed_events()
            
            # Step 4: Calculate quality metrics
            print("📈 Calculating quality metrics...")
            quality_metrics = self._calculate_analysis_quality_metrics(
                tracks, comprehensive_results
            )
            
            # Step 5: Generate match summary
            print("📋 Generating match summary...")
            match_summary = self._generate_enhanced_match_summary(
                comprehensive_results, quality_metrics
            )
            
            # Step 6: Create analysis results
            analysis_results = AnalysisResults(
                timestamp=datetime.now().isoformat(),
                configuration=self.config,
                team_statistics=comprehensive_results.get("team_statistics", {}),
                player_statistics=comprehensive_results.get("player_statistics", {}),
                pass_events=detailed_events["pass_events"],
                tackle_events=detailed_events["tackle_events"],
                interception_events=detailed_events["interception_events"],
                possession_events=detailed_events["possession_events"],
                ball_tracking_quality=comprehensive_results.get("ball_analysis", {}).get("quality_metrics", {}),
                analysis_quality_metrics=quality_metrics,
                frame_coverage=self._calculate_frame_coverage(tracks),
                match_summary=match_summary
            )
            
            # Step 7: Save results if requested
            if save_results:
                print("💾 Saving analysis results...")
                self._save_analysis_results(analysis_results, output_dir)
            
            # Update performance metrics
            self.performance_metrics["analysis_end_time"] = datetime.now()
            self.performance_metrics["processing_time"] = (
                self.performance_metrics["analysis_end_time"] - 
                self.performance_metrics["analysis_start_time"]
            ).total_seconds()
            self.performance_metrics["events_detected"] = (
                len(detailed_events["pass_events"]) + 
                len(detailed_events["tackle_events"]) + 
                len(detailed_events["interception_events"])
            )
            
            # Store current analysis
            self.current_analysis = analysis_results
            self.analysis_history.append(analysis_results)
            
            print("✅ Unified match analysis complete!")
            self._print_analysis_summary(analysis_results)
            
            return analysis_results
            
        except Exception as e:
            print(f"❌ Error during match analysis: {str(e)}")
            raise
    
    def _generate_player_assignments(self, tracks: Dict) -> List[int]:
        """Generate player ball assignments for all frames."""
        player_assignments = []
        
        for frame_num in range(len(tracks.get("ball", []))):
            if (frame_num < len(tracks["players"]) and 
                frame_num < len(tracks["ball"]) and 
                1 in tracks["ball"][frame_num]):
                
                ball_bbox = tracks["ball"][frame_num][1].get("bbox", [])
                if ball_bbox and len(ball_bbox) == 4:
                    assigned_player = self.player_assigner.assign_ball_to_player(
                        tracks["players"][frame_num], ball_bbox
                    )
                    player_assignments.append(assigned_player)
                else:
                    player_assignments.append(-1)
            else:
                player_assignments.append(-1)
        
        return player_assignments
    
    def _extract_detailed_events(self) -> Dict[str, List]:
        """Extract detailed event data from all analyzers."""
        return {
            "pass_events": [asdict(event) for event in self.pass_counter.pass_events],
            "tackle_events": [asdict(event) for event in self.tackle_counter.tackle_events],
            "interception_events": [asdict(event) for event in self.tackle_counter.interception_events],
            "possession_events": [asdict(event) for event in self.pass_counter.possession_events]
        }
    
    def _calculate_analysis_quality_metrics(self, tracks: Dict, 
                                          comprehensive_results: Dict) -> Dict:
        """Calculate quality metrics for the analysis."""
        total_frames = len(tracks.get("ball", []))
        
        # Ball tracking quality
        ball_quality = comprehensive_results.get("ball_analysis", {}).get("quality_metrics", {})
        
        # Event detection quality
        total_events = (
            len(self.pass_counter.pass_events) + 
            len(self.tackle_counter.tackle_events) + 
            len(self.tackle_counter.interception_events)
        )
        
        # Confidence distribution
        all_confidences = []
        for event in self.pass_counter.pass_events:
            all_confidences.append(event.confidence_score)
        for event in self.tackle_counter.tackle_events:
            all_confidences.append(event.confidence_score)
        for event in self.tackle_counter.interception_events:
            all_confidences.append(event.confidence_score)
        
        return {
            "total_frames_analyzed": total_frames,
            "total_events_detected": total_events,
            "events_per_minute": (total_events / (total_frames / self.config.frame_rate)) * 60 if total_frames > 0 else 0,
            "avg_event_confidence": np.mean(all_confidences) if all_confidences else 0,
            "high_confidence_events_ratio": len([c for c in all_confidences if c > 0.7]) / len(all_confidences) if all_confidences else 0,
            "ball_tracking_quality": ball_quality,
            "analysis_completeness": self._calculate_analysis_completeness(tracks),
            "data_consistency_score": self._calculate_data_consistency_score()
        }
    
    def _calculate_frame_coverage(self, tracks: Dict) -> Dict:
        """Calculate frame coverage statistics."""
        total_frames = len(tracks.get("ball", []))
        
        # Ball detection coverage
        ball_detected_frames = sum(1 for frame in tracks.get("ball", []) if 1 in frame and frame[1])
        
        # Player detection coverage
        player_frames_with_detections = sum(1 for frame in tracks.get("players", []) if frame)
        
        return {
            "total_frames": total_frames,
            "ball_detection_coverage": ball_detected_frames / total_frames if total_frames > 0 else 0,
            "player_detection_coverage": player_frames_with_detections / total_frames if total_frames > 0 else 0,
            "complete_frame_coverage": min(ball_detected_frames, player_frames_with_detections) / total_frames if total_frames > 0 else 0
        }
    
    def _calculate_analysis_completeness(self, tracks: Dict) -> float:
        """Calculate overall analysis completeness score."""
        frame_coverage = self._calculate_frame_coverage(tracks)
        
        # Weight different aspects of completeness
        ball_weight = 0.4
        player_weight = 0.3
        event_weight = 0.3
        
        ball_score = frame_coverage["ball_detection_coverage"]
        player_score = frame_coverage["player_detection_coverage"]
        
        # Event detection score based on reasonable expectations
        total_frames = len(tracks.get("ball", []))
        expected_events_per_minute = 15  # Reasonable expectation
        expected_total_events = (total_frames / self.config.frame_rate / 60) * expected_events_per_minute
        
        actual_events = (
            len(self.pass_counter.pass_events) + 
            len(self.tackle_counter.tackle_events) + 
            len(self.tackle_counter.interception_events)
        )
        
        event_score = min(1.0, actual_events / max(1, expected_total_events))
        
        return ball_score * ball_weight + player_score * player_weight + event_score * event_weight
    
    def _calculate_data_consistency_score(self) -> float:
        """Calculate data consistency score across different analyzers."""
        # This is a simplified consistency check
        # In a full implementation, this would check for consistency between
        # different analyzers' results
        
        pass_events_count = len(self.pass_counter.pass_events)
        possession_events_count = len(self.pass_counter.possession_events)
        
        # Basic consistency: should have more possessions than passes
        if possession_events_count == 0:
            return 0.5  # Neutral score if no possession data
        
        ratio = pass_events_count / possession_events_count
        # Expect roughly 1-3 passes per possession on average
        if 0.5 <= ratio <= 3.0:
            return 1.0  # Good consistency
        elif 0.2 <= ratio <= 5.0:
            return 0.7  # Acceptable consistency
        else:
            return 0.3  # Poor consistency
    
    def _generate_enhanced_match_summary(self, comprehensive_results: Dict, 
                                       quality_metrics: Dict) -> Dict:
        """Generate enhanced match summary with quality indicators."""
        base_summary = comprehensive_results.get("match_overview", {})
        
        # Add quality and performance metrics
        enhanced_summary = {
            **base_summary,
            "analysis_quality": {
                "overall_quality_score": quality_metrics.get("analysis_completeness", 0),
                "data_consistency_score": quality_metrics.get("data_consistency_score", 0),
                "avg_event_confidence": quality_metrics.get("avg_event_confidence", 0),
                "high_confidence_events_ratio": quality_metrics.get("high_confidence_events_ratio", 0)
            },
            "performance_metrics": self.performance_metrics,
            "enhanced_features_used": {
                "enhanced_ball_tracking": self.config.enable_enhanced_ball_tracking,
                "trajectory_analysis": self.config.enable_trajectory_analysis,
                "confidence_validation": self.config.enable_confidence_validation
            }
        }
        
        return enhanced_summary
    
    def _save_analysis_results(self, results: AnalysisResults, output_dir: str):
        """Save comprehensive analysis results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main results as JSON
        results_path = os.path.join(output_dir, "unified_analysis_results.json")
        with open(results_path, 'w') as f:
            # Convert dataclass to dict for JSON serialization
            results_dict = asdict(results)
            json.dump(results_dict, f, indent=2, default=str)
        
        # Save as pickle for complete object preservation
        pickle_path = os.path.join(output_dir, "unified_analysis_results.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Export CSV files using component analyzers
        if self.config.export_detailed_events:
            self.pass_counter.export_enhanced_statistics_to_csv(output_dir)
            self.tackle_counter.export_enhanced_statistics_to_csv(output_dir)
        
        if self.config.export_team_stats or self.config.export_player_stats:
            self.comprehensive_stats.export_comprehensive_statistics(output_dir)
        
        # Save configuration
        config_path = os.path.join(output_dir, "analysis_configuration.json")
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        print(f"📁 Analysis results saved to: {output_dir}")
    
    def _print_analysis_summary(self, results: AnalysisResults):
        """Print a summary of the analysis results."""
        print("\n" + "="*60)
        print("📊 UNIFIED MATCH ANALYSIS SUMMARY")
        print("="*60)
        
        # Basic statistics
        team_stats = results.team_statistics
        if team_stats:
            print(f"\n🏆 TEAM STATISTICS:")
            for team_id, stats in team_stats.items():
                if hasattr(stats, 'possession_percentage'):
                    print(f"  Team {team_id}: {stats.possession_percentage:.1f}% possession, "
                          f"{stats.total_passes} passes, {stats.total_tackles} tackles")
        
        # Event summary
        print(f"\n⚽ EVENTS DETECTED:")
        print(f"  • Passes: {len(results.pass_events)}")
        print(f"  • Tackles: {len(results.tackle_events)}")
        print(f"  • Interceptions: {len(results.interception_events)}")
        print(f"  • Possessions: {len(results.possession_events)}")
        
        # Quality metrics
        quality = results.analysis_quality_metrics
        print(f"\n📈 ANALYSIS QUALITY:")
        print(f"  • Overall Quality: {quality.get('analysis_completeness', 0):.2f}")
        print(f"  • Data Consistency: {quality.get('data_consistency_score', 0):.2f}")
        print(f"  • Avg Event Confidence: {quality.get('avg_event_confidence', 0):.2f}")
        
        # Performance
        perf = self.performance_metrics
        print(f"\n⚡ PERFORMANCE:")
        print(f"  • Processing Time: {perf['processing_time']:.1f}s")
        print(f"  • Frames Processed: {perf['frames_processed']}")
        print(f"  • Events per Second: {perf['events_detected'] / max(1, perf['processing_time']):.1f}")
        
        print("="*60)
    
    def get_analysis_summary(self) -> Optional[Dict]:
        """Get a summary of the current analysis."""
        if not self.current_analysis:
            return None
        
        return {
            "timestamp": self.current_analysis.timestamp,
            "team_statistics": self.current_analysis.team_statistics,
            "total_events": (
                len(self.current_analysis.pass_events) + 
                len(self.current_analysis.tackle_events) + 
                len(self.current_analysis.interception_events)
            ),
            "quality_score": self.current_analysis.analysis_quality_metrics.get("analysis_completeness", 0),
            "performance_metrics": self.performance_metrics
        }
    
    def export_summary_report(self, output_path: str = "data/output/analysis_summary.txt"):
        """Export a human-readable summary report."""
        if not self.current_analysis:
            print("No analysis results available to export.")
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("FOOTBALL MATCH ANALYSIS REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Analysis Date: {self.current_analysis.timestamp}\n")
            f.write(f"Configuration: Enhanced Ball Tracking Enabled\n\n")
            
            # Team statistics
            f.write("TEAM PERFORMANCE\n")
            f.write("-"*20 + "\n")
            for team_id, stats in self.current_analysis.team_statistics.items():
                if hasattr(stats, 'possession_percentage'):
                    f.write(f"Team {team_id}:\n")
                    f.write(f"  Possession: {stats.possession_percentage:.1f}%\n")
                    f.write(f"  Passes: {stats.total_passes} (Accuracy: {stats.pass_accuracy:.1%})\n")
                    f.write(f"  Tackles: {stats.total_tackles} (Success: {stats.tackle_success_rate:.1%})\n")
                    f.write(f"  Interceptions: {stats.total_interceptions}\n\n")
            
            # Quality metrics
            quality = self.current_analysis.analysis_quality_metrics
            f.write("ANALYSIS QUALITY\n")
            f.write("-"*20 + "\n")
            f.write(f"Overall Quality Score: {quality.get('analysis_completeness', 0):.2f}/1.00\n")
            f.write(f"Data Consistency: {quality.get('data_consistency_score', 0):.2f}/1.00\n")
            f.write(f"Average Event Confidence: {quality.get('avg_event_confidence', 0):.2f}/1.00\n\n")
            
            # Performance
            f.write("PROCESSING PERFORMANCE\n")
            f.write("-"*20 + "\n")
            f.write(f"Processing Time: {self.performance_metrics['processing_time']:.1f} seconds\n")
            f.write(f"Frames Analyzed: {self.performance_metrics['frames_processed']}\n")
            f.write(f"Events Detected: {self.performance_metrics['events_detected']}\n")
        
        print(f"📄 Summary report exported to: {output_path}")
