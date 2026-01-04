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
    
    # ปรับปรุงความชัดเจนของกราฟ
    fig.update_layout(
        font=dict(size=14, family="Arial, sans-serif"),  # เพิ่มขนาดฟอนต์และกำหนดฟอนต์
        title_font=dict(size=18, family="Arial, sans-serif"),  # ขนาดฟอนต์ชื่อกราฟ
        xaxis=dict(
            titlefont=dict(size=16),  # ขนาดฟอนต์ชื่อแกน X
            tickfont=dict(size=13)    # ขนาดฟอนต์ตัวเลขแกน X
        ),
        yaxis=dict(
            titlefont=dict(size=16),  # ขนาดฟอนต์ชื่อแกน Y
            tickfont=dict(size=13)    # ขนาดฟอนต์ข้อความแกน Y
        ),
        hoverlabel=dict(
            font_size=14,  # ขนาดฟอนต์ใน hover tooltip
            font_family="Arial, sans-serif"
        )
    )
    
    # ปรับขนาดเส้นและจุดให้เห็นชัดขึ้น
    fig.update_traces(
        line=dict(width=3),  # ความหนาของเส้น
        marker=dict(size=10)  # ขนาดจุด
    )
    
    return fig.to_html(full_html=False)  # Return HTML string