import os
import sys
from openai import OpenAI
from time_series_analysis import analyze_video_over_time
from interactive_visualization import create_interactive_dashboard, generate_video_with_overlay


def setup_openai_client():
    """Setup the OpenAI client with API key."""
    # Check if API key is already set
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        import getpass
        api_key = getpass.getpass("Enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key

    # Initialize OpenAI client
    client = OpenAI()

    # Test the client with a simple request
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'OpenAI connection successful'"}
            ],
            max_tokens=10
        )
        print("OpenAI client setup complete.")
    except Exception as e:
        print(f"Error setting up OpenAI client: {e}")
        sys.exit(1)

    return client


def run_analysis(video_path, output_dir="output", segment_length=5.0, generate_overlay=True):
    """
    Run the complete analysis pipeline on a video file.

    Args:
        video_path: Path to the video file to analyze
        output_dir: Directory to store output files
        segment_length: Length of audio segments for analysis (in seconds)
        generate_overlay: Whether to generate a video with analysis overlay

    Returns:
        Dictionary containing paths to generated files and analysis results
    """
    # Setup OpenAI client only
    client = setup_openai_client()

    print(f"\n{'='*50}")
    print(f"Starting analysis of: {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"Segment length: {segment_length} seconds")
    print(f"{'='*50}\n")

    print(f"\n{'='*50}")
    print(f"Starting analysis of: {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"Segment length: {segment_length} seconds")
    print(f"{'='*50}\n")

    # Step 1: Run time-based analysis
    print("Running time-based analysis...")
    analysis_results = analyze_video_over_time(
        video_path, segment_length, output_dir)

    # Extract the data frames from the results
    text_df = analysis_results['data']['text_df']
    audio_df = analysis_results['data']['audio_df']
    emotion_df = analysis_results['data']['emotion_df']
    mesh_df = analysis_results['data']['mesh_df']
    insights_df = analysis_results['data']['insights_df']

    # Step 2: Create interactive visualization
    print("Creating interactive visualization...")
    dashboard_path, interactive_report_path = create_interactive_dashboard(
        text_df, audio_df, emotion_df, mesh_df, insights_df, video_path, output_dir
    )

    # Step 3: Generate video with overlay (optional)
    overlay_path = None
    if generate_overlay:
        print("Generating video with analysis overlay...")
        try:
            overlay_path = generate_video_with_overlay(
                video_path, text_df, emotion_df, audio_df, mesh_df, output_dir
            )
            print(f"Video with overlay saved to: {overlay_path}")
        except Exception as e:
            print(f"Error generating video with overlay: {e}")
            print("Continuing without overlay video...")

    # Combine all results
    results = {
        "static_dashboard_path": analysis_results['dashboard_path'],
        "static_report_path": analysis_results['report_path'],
        "interactive_dashboard_path": dashboard_path,
        "interactive_report_path": interactive_report_path,
        "video_overlay_path": overlay_path,
        "report_text": analysis_results['report_text'],
        "data": {
            "text_df": text_df,
            "audio_df": audio_df,
            "emotion_df": emotion_df,
            "mesh_df": mesh_df,
            "insights_df": insights_df
        }
    }

    print(f"\n{'='*50}")
    print("Analysis complete!")
    print(f"Static dashboard: {results['static_dashboard_path']}")
    print(f"Static report: {results['static_report_path']}")
    print(f"Interactive dashboard: {results['interactive_dashboard_path']}")
    print(f"Interactive report: {results['interactive_report_path']}")
    if overlay_path:
        print(f"Video with overlay: {results['video_overlay_path']}")
    print(f"{'='*50}\n")

    print("Executive Summary:")
    print(results['report_text'])

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Time-based multimodal analysis of video communication")
    parser.add_argument("video_path", help="Path to the video file to analyze")
    parser.add_argument("--output-dir", default="output",
                        help="Directory to store output files")
    parser.add_argument("--segment-length", type=float, default=5.0,
                        help="Length of audio segments for analysis (in seconds)")
    parser.add_argument("--no-overlay", action="store_true",
                        help="Skip generation of video with analysis overlay")

    args = parser.parse_args()

    run_analysis(
        video_path=args.video_path,
        output_dir=args.output_dir,
        segment_length=args.segment_length,
        generate_overlay=not args.no_overlay
    )
