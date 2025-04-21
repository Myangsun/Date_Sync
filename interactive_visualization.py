import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.offline import plot
import base64
from datetime import datetime
from moviepy import VideoFileClip, TextClip, CompositeVideoClip, ColorClip
import matplotlib.colors as mcolors


def create_interactive_dashboard(text_df, audio_df, emotion_df, mesh_df, insights_df, video_path, output_dir="output"):
    """
    Create an interactive HTML dashboard for time-based multimodal analysis results.
    """
    # Create output directory for interactive visualizations
    interactive_dir = os.path.join(output_dir, "interactive")
    os.makedirs(interactive_dir, exist_ok=True)

    # Create a subplot figure
    fig = sp.make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            "Speech Sentiment",
            "Voice Characteristics",
            "Facial Emotions",
            "Eye Fixation Variance (Engagement)",
            "Facial Comfort Features"
        ),
        row_heights=[0.15, 0.15, 0.25, 0.15, 0.3]
    )

    # 1. Plot text sentiment over time
    fig.add_trace(
        go.Scatter(
            x=text_df['time'],
            y=text_df['sentiment'],
            mode='lines+markers',
            name='Sentiment',
            line=dict(color='blue'),
            hovertemplate='Time: %{x:.1f}s<br>Sentiment: %{y:.2f}<br>Text: %{text}',
            text=text_df['text']
        ),
        row=1, col=1
    )

    # Add key moments as markers
    if not insights_df.empty:
        fig.add_trace(
            go.Scatter(
                x=insights_df['time'],
                y=insights_df['sentiment'],
                mode='markers',
                marker=dict(
                    size=12,
                    symbol='star',
                    color='red',
                    line=dict(width=2, color='black')
                ),
                name='Key Moments',
                hovertemplate='Time: %{x:.1f}s<br>Insight: %{text}',
                text=insights_df['insight']
            ),
            row=1, col=1
        )

    # 2. Plot audio features over time
    fig.add_trace(
        go.Scatter(
            x=audio_df['time'],
            y=audio_df['pitch'],
            mode='lines+markers',
            name='Pitch (Hz)',
            line=dict(color='blue'),
            hovertemplate='Time: %{x:.1f}s<br>Pitch: %{y:.1f} Hz'
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=audio_df['time'],
            y=audio_df['intensity'],
            mode='lines+markers',
            name='Intensity (dB)',
            line=dict(color='red'),
            hovertemplate='Time: %{x:.1f}s<br>Intensity: %{y:.1f} dB',
            yaxis='y2'
        ),
        row=2, col=1
    )

    # 3. Plot emotions over time
    emotions = ['happy', 'sad', 'angry', 'fear',
                'disgust', 'surprise', 'neutral']
    colors = ['green', 'blue', 'red', 'purple', 'brown', 'orange', 'gray']

    for emotion, color in zip(emotions, colors):
        fig.add_trace(
            go.Scatter(
                x=emotion_df['time'],
                y=emotion_df[emotion],
                mode='lines',
                name=emotion.capitalize(),
                line=dict(color=color),
                hovertemplate=f'Time: %{{x:.1f}}s<br>{emotion.capitalize()}: %{{y:.2f}}'
            ),
            row=3, col=1
        )

    # 4. Plot eye fixation over time
    fig.add_trace(
        go.Scatter(
            x=mesh_df['time'],
            y=mesh_df['eye_fixation_var'],
            mode='lines',
            name='Eye Fixation Var',
            line=dict(color='purple'),
            hovertemplate='Time: %{x:.1f}s<br>Eye Fixation Var: %{y:.4f}'
        ),
        row=4, col=1
    )

    # 5. Plot facial comfort features over time
    fig.add_trace(
        go.Scatter(
            x=mesh_df['time'],
            y=mesh_df['frown'],
            mode='lines',
            name='Frown',
            line=dict(color='red'),
            hovertemplate='Time: %{x:.1f}s<br>Frown: %{y:.4f}'
        ),
        row=5, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=mesh_df['time'],
            y=mesh_df['jaw'],
            mode='lines',
            name='Jaw',
            line=dict(color='blue'),
            hovertemplate='Time: %{x:.1f}s<br>Jaw: %{y:.4f}'
        ),
        row=5, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=mesh_df['time'],
            y=mesh_df['mouth_open'],
            mode='lines',
            name='Mouth Open',
            line=dict(color='green'),
            hovertemplate='Time: %{x:.1f}s<br>Mouth Open: %{y:.4f}'
        ),
        row=5, col=1
    )

    # Add a slider for time navigation
    fig.update_layout(
        title={
            'text': 'Multimodal Communication Analysis Over Time',
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='closest',
        height=900,
        width=1200,
        template='plotly_white',
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 20},
                'prefix': 'Time: ',
                'suffix': ' seconds',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 300, 'easing': 'cubic-in-out'},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': [
                {
                    'args': [
                        {'x': [[t], [t], [t], [t], [t], [t],
                               [t], [t], [t], [t], [t], [t]]},
                        {'mode': 'lines+markers', 'frame': {'duration': 500,
                                                            'redraw': False}, 'transition': {'duration': 300}}
                    ],
                    'label': f"{t:.0f}",
                    'method': 'animate'
                } for t in range(0, int(max(mesh_df['time'])) + 1, 5)  # Create steps every 5 seconds
            ]
        }]
    )

    # Update y-axis titles
    fig.update_yaxes(title_text="Sentiment (-1 to 1)", row=1, col=1)
    fig.update_yaxes(title_text="Pitch (Hz)", row=2, col=1)
    fig.update_yaxes(title_text="Intensity (dB)",
                     overlaying="y", side="right", row=2, col=1)
    fig.update_yaxes(title_text="Probability", row=3, col=1)
    fig.update_yaxes(title_text="Variance", row=4, col=1)
    fig.update_yaxes(title_text="Value", row=5, col=1)

    # Update x-axis title for the bottom subplot only
    fig.update_xaxes(title_text="Time (seconds)", row=5, col=1)

    # Save the interactive plot
    dashboard_path = os.path.join(
        interactive_dir, 'interactive_dashboard.html')
    plot(fig, filename=dashboard_path, auto_open=False)

    # Create HTML report with integrated interactive dashboard
    # Generate a detailed table of key moments
    moments_table = ""
    if not insights_df.empty:
        moments_table = """
        <table class="insights-table">
            <thead>
                <tr>
                    <th>Time (s)</th>
                    <th>Text</th>
                    <th>Emotion</th>
                    <th>Analysis</th>
                </tr>
            </thead>
            <tbody>
        """

        for _, row in insights_df.iterrows():
            moments_table += f"""
                <tr>
                    <td>{row['time']:.1f}</td>
                    <td>"{row['text']}"</td>
                    <td>{row['dominant_emotion']}</td>
                    <td>{row['insight']}</td>
                </tr>
            """

        moments_table += """
            </tbody>
        </table>
        """

    # Create the full HTML report with the embedded interactive dashboard
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Interactive Multimodal Communication Analysis</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f9f9f9;
            }}
            header {{
                background-color: #2c3e50;
                color: white;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
                text-align: center;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .dashboard {{
                margin: 30px 0;
                padding: 15px;
                background: white;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .dashboard iframe {{
                width: 100%;
                height: 900px;
                border: none;
            }}
            .insights-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            .insights-table th, .insights-table td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            .insights-table th {{
                background-color: #2c3e50;
                color: white;
            }}
            .insights-table tr:hover {{
                background-color: #f5f5f5;
            }}
            .timestamp {{
                font-weight: bold;
                color: #2980b9;
            }}
            footer {{
                margin-top: 50px;
                text-align: center;
                color: #7f8c8d;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <header>
            <h1>Interactive Multimodal Communication Analysis</h1>
            <p>Video: {os.path.basename(video_path)}</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>
        
        <section class="dashboard">
            <h2>Interactive Dashboard</h2>
            <p>Use the time slider at the bottom of the chart to navigate through the video timeline. Hover over data points for detailed information.</p>
            <iframe src="{os.path.basename(dashboard_path)}"></iframe>
        </section>
        
        <section class="key-moments">
            <h2>Key Communication Moments</h2>
            <p>The following table shows key moments identified in the communication with detailed analysis.</p>
            {moments_table}
        </section>
        
        <footer>
            <p>Powered by OpenAI, Plotly, and MediaPipe</p>
        </footer>
    </body>
    </html>
    """

    # Save the HTML report
    report_path = os.path.join(interactive_dir, 'interactive_report.html')
    with open(report_path, 'w') as f:
        f.write(html_content)

    return dashboard_path, report_path


def generate_video_with_overlay(video_path, text_df, emotion_df, audio_df, mesh_df, output_dir="output"):
    """
    Generate a video with real-time analysis overlay showing key metrics as the video plays.
    This requires moviepy for video processing.
    """
    # Create output directory
    overlay_dir = os.path.join(output_dir, "overlay")
    os.makedirs(overlay_dir, exist_ok=True)

    # Load the video
    video = VideoFileClip(video_path)

    # Function to create text overlay for a specific time
    def make_text_overlay(t):
        # Find the closest data points to the current time
        text_idx = (text_df['time'] -
                    t).abs().idxmin() if not text_df.empty else None
        emotion_idx = (
            emotion_df['time'] - t).abs().idxmin() if not emotion_df.empty else None
        audio_idx = (audio_df['time'] -
                     t).abs().idxmin() if not audio_df.empty else None
        mesh_idx = (mesh_df['time'] -
                    t).abs().idxmin() if not mesh_df.empty else None

        text_row = text_df.iloc[text_idx] if text_idx is not None else None
        emotion_row = emotion_df.iloc[emotion_idx] if emotion_idx is not None else None
        audio_row = audio_df.iloc[audio_idx] if audio_idx is not None else None
        mesh_row = mesh_df.iloc[mesh_idx] if mesh_idx is not None else None

        # Prepare text overlay
        overlay_text = f"Time: {t:.1f}s\n"

        # Add sentiment if available
        if text_row is not None:
            sentiment = text_row['sentiment']
            sentiment_color = "green" if sentiment > 0 else "red"
            overlay_text += f"Sentiment: {sentiment:.2f}\n"

        # Add dominant emotion if available
        if emotion_row is not None:
            emotions = ['angry', 'disgust', 'fear',
                        'happy', 'sad', 'surprise', 'neutral']
            dominant_emotion = emotions[np.argmax(
                emotion_row[emotions].values)]
            overlay_text += f"Emotion: {dominant_emotion.capitalize()}\n"

        # Add pitch and intensity if available
        if audio_row is not None:
            overlay_text += f"Pitch: {audio_row['pitch']:.1f} Hz\n"
            overlay_text += f"Volume: {audio_row['intensity']:.1f} dB\n"

        # Add engagement if available
        if mesh_row is not None and 'eye_fixation_var' in mesh_row:
            engagement = "High" if mesh_row['eye_fixation_var'] < 0.0005 else "Medium" if mesh_row['eye_fixation_var'] < 0.001 else "Low"
            overlay_text += f"Engagement: {engagement}\n"

        # Create text clip
        txt_clip = TextClip(
            overlay_text,
            fontsize=20,
            color='white',
            bg_color='black',
            font='Arial-Bold',
            kerning=2,
            method='caption',
            align='West',
            size=(300, None)
        )

        # Position in the top-right corner
        txt_clip = txt_clip.set_position(('right', 'top'))

        # Create color bar for sentiment
        if text_row is not None:
            sentiment = text_row['sentiment']
            # Map sentiment from [-1, 1] to [0, 1] for color gradient
            norm_sentiment = (sentiment + 1) / 2
            color = list(mcolors.hsv_to_rgb([0.3 * norm_sentiment, 0.9, 0.9]))
            color.append(1.0)  # Add alpha

            sentiment_bar = ColorClip((150, 10), col=color)
            sentiment_bar = sentiment_bar.set_position(
                (video.w - 300, txt_clip.h + 10))
            return [txt_clip, sentiment_bar]

        return [txt_clip]

    # Create composite video
    def make_frame(t):
        frame = video.get_frame(t)

        # Create a base clip with the original frame
        base_clip = ColorClip(
            size=(frame.shape[1], frame.shape[0]), color=(0, 0, 0, 0))
        base_clip.get_frame = lambda t: frame

        # Create overlays
        overlays = make_text_overlay(t)

        # Create a composite clip with the base clip and overlays
        clips = [base_clip] + overlays
        composite = CompositeVideoClip(clips, size=base_clip.size)

        return composite.get_frame(0)

    # Create video with overlay
    video_with_overlay = VideoClip(make_frame, duration=video.duration)
    video_with_overlay = video_with_overlay.set_audio(video.audio)

    # Write to file
    output_path = os.path.join(
        overlay_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_with_overlay.mp4")
    video_with_overlay.write_videofile(
        output_path, codec='libx264', audio_codec='aac', fps=video.fps)

    video.close()

    return output_path
