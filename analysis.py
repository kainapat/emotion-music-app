import plotly.express as px
import pandas as pd

def build_trajectory(segments, emotions):
    return [(i, e) for i, e in enumerate(emotions)]

def plot_interactive_trajectory(emotions, song_name):
    df = pd.DataFrame({"step": range(len(emotions)), "emotion": emotions})
    fig = px.line(
        df, x="step", y="emotion",
        title=f"Emotion Trajectory: {song_name}",
        markers=True,
        labels={
            "step": "Step",  # X-axis label in English
            "emotion": "Emotion"  # Y-axis label in English
        }
    )
    return fig.to_html(full_html=False)  # Return HTML string