import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import textwrap
import os
import re
import io
import zipfile
import numpy as np
from datetime import datetime

# --- Configuration ---
METADATA_END_INDEX = 8
OUTPUT_DIR = "Teacher_Output"

# Set page config for premium look
st.set_page_config(
    page_title="Teacher Feedback Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling
st.markdown("""
<style>
    /* Main container styling */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Chart container */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #e1e5e9;
    }
    
    /* File uploader styling */
    .stFileUploader > div > div {
        border-radius: 10px;
        border: 2px dashed #667eea;
        background-color: #f8f9ff;
    }
    
    /* Success/Info messages */
    .stSuccess, .stInfo {
        border-radius: 10px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 10px;
        background-color: #f1f3f6;
        color: #666;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Utilities ---
def create_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def sanitize_filename(name):
    return re.sub(r'[\\/:"*?<>|]+', '_', name).strip()

def calculate_cumulative_percentage(series):
    valid = series.dropna()
    score = valid.mean() if len(valid) > 0 else 0
    return (score / 5) * 100

def get_color_palette(palette_name, n_colors=5):
    """Get color palette based on selection"""
    palettes = {
        "Modern Blues": ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c"],
        "Sunset": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"],
        "Professional": ["#2C3E50", "#34495E", "#7F8C8D", "#BDC3C7", "#ECF0F1"],
        "Vibrant": ["#E74C3C", "#F39C12", "#F1C40F", "#2ECC71", "#3498DB"],
        "Purple Haze": ["#8E44AD", "#9B59B6", "#BB8FCE", "#D2B4DE", "#E8DAEF"],
        "Ocean": ["#006994", "#13A3D1", "#70C7E3", "#B8E0F0", "#E1F4FA"]
    }
    return palettes.get(palette_name, palettes["Modern Blues"])

# --- Enhanced Plotting Functions ---
def plot_distribution_enhanced(df_long, course, chart_params):
    """Enhanced distribution plot with multiple chart types"""
    count_df = df_long.groupby(['Question','Response']).size().reset_index(name='Count')
    total_per_question = count_df.groupby('Question')['Count'].transform('sum')
    count_df['Percentage'] = count_df['Count'] / total_per_question * 100
    questions = list(pd.Categorical(count_df['Question']).categories)
    count_df['Question'] = pd.Categorical(count_df['Question'], categories=questions, ordered=True)

    if chart_params['chart_type'] == 'Plotly Interactive':
        fig = px.bar(
            count_df, 
            x='Question', 
            y='Percentage', 
            color='Response',
            title=chart_params['title'],
            color_discrete_sequence=get_color_palette(chart_params['color_palette']),
            height=chart_params['height'] * 50
        )
        
        fig.update_layout(
            font=dict(size=chart_params['font_size']),
            title_font_size=chart_params['title_font'],
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        return fig
    
    else:  # Matplotlib charts
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        fig, ax = plt.subplots(figsize=(chart_params['width'], chart_params['height']))
        
        if chart_params['chart_type'] == 'Bar Chart':
            bars = sns.barplot(
                data=count_df,
                x='Question',
                y='Percentage',
                hue='Response',
                palette=get_color_palette(chart_params['color_palette']),
                ax=ax,
                order=questions
            )
            
            # Enhance bar appearance
            for patch in bars.patches:
                patch.set_linewidth(chart_params['bar_thickness'])
                patch.set_edgecolor('white')
                
        elif chart_params['chart_type'] == 'Histogram':
            # Convert to histogram-like representation
            for i, response in enumerate(sorted(count_df['Response'].unique())):
                response_data = count_df[count_df['Response'] == response]
                ax.bar(
                    response_data['Question'], 
                    response_data['Percentage'],
                    alpha=0.7,
                    width=chart_params['bar_thickness'],
                    label=f'Response {response}',
                    color=get_color_palette(chart_params['color_palette'])[i]
                )
        
        # Styling
        wrapped_labels = ["\n".join(textwrap.wrap(str(q), 25)) for q in questions]
        ax.set_xticklabels(wrapped_labels, rotation=chart_params['x_rotation'], 
                          ha='center', fontsize=chart_params['font_size'])
        ax.set_xlabel(chart_params['x_label'], fontsize=chart_params['font_size'] + 2)
        ax.set_ylabel(chart_params['y_label'], fontsize=chart_params['font_size'] + 2)
        ax.set_title(chart_params['title'], fontsize=chart_params['title_font'], 
                    pad=20, fontweight='bold')
        ax.set_ylim(0, chart_params['y_max'])
        
        # Grid and styling
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#fafafa')
        
        if chart_params['show_legend']:
            ax.legend(title='Response (1-5)', bbox_to_anchor=(1.02, 0.5), 
                     loc='center left', frameon=True, shadow=True)
        else:
            ax.get_legend().remove() if ax.get_legend() else None
            
        plt.tight_layout()
        st.pyplot(fig)
        return fig

def plot_average_scores_enhanced(mean_scores, course, chart_params):
    """Enhanced average scores plot"""
    if chart_params['chart_type'] == 'Plotly Interactive':
        fig = px.bar(
            x=mean_scores.values,
            y=mean_scores.index,
            orientation='h',
            title=chart_params['title'],
            color=mean_scores.values,
            color_continuous_scale=chart_params['color_palette'].lower(),
            height=chart_params['height'] * 50
        )
        
        fig.update_layout(
            font=dict(size=chart_params['font_size']),
            title_font_size=chart_params['title_font'],
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='rgba(128,128,128,0.2)', range=[1, 5]),
            yaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
            showlegend=chart_params['show_legend']
        )
        
        st.plotly_chart(fig, use_container_width=True)
        return fig
    
    else:  # Matplotlib
        fig, ax = plt.subplots(figsize=(chart_params['width'], chart_params['height']))
        
        bars = ax.barh(
            mean_scores.index,
            mean_scores.values,
            color=get_color_palette(chart_params['color_palette'])[:len(mean_scores)],
            edgecolor='white',
            linewidth=chart_params['bar_thickness']
        )
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.05, bar.get_y() + bar.get_height()/2, 
                   f'{width:.2f}', ha='left', va='center', 
                   fontsize=chart_params['font_size'], fontweight='bold')
        
        ax.set_xlim(1, 5.5)
        ax.set_xlabel(chart_params['x_label'], fontsize=chart_params['font_size'] + 2)
        ax.set_ylabel(chart_params['y_label'], fontsize=chart_params['font_size'] + 2)
        ax.set_title(chart_params['title'], fontsize=chart_params['title_font'], 
                    pad=20, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_facecolor('#fafafa')
        
        plt.tight_layout()
        st.pyplot(fig)
        return fig

def plot_cumulative_pie_enhanced(course, percent, chart_params):
    """Enhanced pie chart with donut style"""
    if chart_params['chart_type'] == 'Plotly Interactive':
        fig = go.Figure(data=[go.Pie(
            labels=['Score', 'Remaining'],
            values=[percent, 100 - percent],
            hole=chart_params['donut_width'],
            marker_colors=[chart_params['color_main'], chart_params['color_bg']],
            textinfo='none'
        )])
        
        fig.add_annotation(
            text=f"{percent:.1f}%<br>Mean Score",
            x=0.5, y=0.5,
            font_size=chart_params['pct_font'],
            showarrow=False
        )
        
        fig.update_layout(
            title=dict(
                text=chart_params['title'],
                font=dict(size=chart_params['title_font']),
                x=0.5
            ),
            height=chart_params['height'] * 50,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        return fig
    
    else:  # Matplotlib
        score = max(min(percent, 100), 0)
        remainder = 100 - score
        fig, ax = plt.subplots(figsize=(chart_params['width'], chart_params['height']))
        
        wedges, texts = ax.pie(
            [score, remainder],
            labels=['', ''],
            colors=[chart_params['color_main'], chart_params['color_bg']],
            startangle=90,
            wedgeprops={'width': chart_params['donut_width'], 'edgecolor': 'white', 'linewidth': 2}
        )

        if chart_params['show_pct']:
            ax.text(0, 0.05, f'{score:.1f}%', ha='center', va='center',
                    fontsize=chart_params['pct_font'], fontweight='bold', color='black')
            ax.text(0, -0.18, 'Mean Score', ha='center', va='center', 
                   fontsize=chart_params['pct_font']-6, color='gray')

        ax.set_title(chart_params['title'], fontsize=chart_params['title_font'], 
                    pad=20, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        return fig

# --- Enhanced Course Processing ---
def process_course_enhanced(df, course, feedback_cols, ui_params):
    """Enhanced course processing with premium UI elements"""
    
    # Course header with metrics
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="margin: 0; color: #667eea;">📚 {course}</h2>
            <p style="margin: 5px 0 0 0; color: #666;">Detailed Analytics Dashboard</p>
        </div>
        """, unsafe_allow_html=True)
    
    df_course = df[df['COURSE'] == course].copy()
    if df_course.empty:
        st.warning(f"⚠️ No data available for {course}")
        return

    current_cols = [col for col in feedback_cols if col in df_course.columns]
    if not current_cols:
        st.warning(f"⚠️ No feedback columns found for {course}")
        return

    df_numeric = df_course[current_cols].apply(pd.to_numeric, errors='coerce')
    
    # Display key metrics
    with col2:
        total_responses = df_numeric.count().sum()
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: #2ecc71;">📊 {total_responses}</h3>
            <p style="margin: 5px 0 0 0; color: #666;">Total Responses</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_score = df_numeric.mean().mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: #e74c3c;">⭐ {avg_score:.2f}</h3>
            <p style="margin: 5px 0 0 0; color: #666;">Overall Average</p>
        </div>
        """, unsafe_allow_html=True)

    df_long = df_numeric.melt(var_name='Question', value_name='Response').dropna()

    if not df_long.empty:
        # Create tabs for different chart types
        tab1, tab2, tab3 = st.tabs(["📊 Distribution Analysis", "📈 Average Scores", "🎯 Overall Performance"])
        
        with tab1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            plot_distribution_enhanced(df_long, course, ui_params['dist'])
            st.markdown('</div>', unsafe_allow_html=True)

        with tab2:
            mean_scores = df_numeric.mean().sort_values()
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            plot_average_scores_enhanced(mean_scores, course, ui_params['avg'])
            st.markdown('</div>', unsafe_allow_html=True)

        with tab3:
            flat = df_numeric.values.flatten()
            pct = calculate_cumulative_percentage(pd.Series(flat))
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                plot_cumulative_pie_enhanced(course, pct, ui_params['pie'])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 📋 Performance Summary")
                performance_level = "Excellent" if pct >= 80 else "Good" if pct >= 60 else "Needs Improvement"
                color = "#2ecc71" if pct >= 80 else "#f39c12" if pct >= 60 else "#e74c3c"
                
                st.markdown(f"""
                <div style="background: {color}; color: white; padding: 1rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;">
                    <h3 style="margin: 0;">{performance_level}</h3>
                    <p style="margin: 5px 0 0 0;">Performance Level</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional statistics
                st.metric("Response Rate", f"{(df_numeric.notna().sum().sum() / (len(df_numeric) * len(current_cols)) * 100):.1f}%")
                st.metric("Questions Analyzed", len(current_cols))
                st.metric("Standard Deviation", f"{df_numeric.std().mean():.2f}")
                
    else:
        st.warning(f"⚠️ No valid numeric responses found for {course}")

# --- Main Streamlit Application ---
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem;">📊 Teacher Feedback Analytics</h1>
        <p style="margin: 10px 0 0 0; font-size: 1.2rem; opacity: 0.9;">Professional Data Visualization & Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # File upload section
    uploaded_file = st.file_uploader(
        "📁 Upload your CSV file", 
        type=["csv"],
        help="Upload a CSV file containing teacher feedback data with a 'COURSE' column"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(df)} records!")
        except Exception as e:
            st.error(f"❌ Error loading CSV: {e}")
            st.stop()

        if 'COURSE' not in df.columns:
            st.error("❌ CSV must contain a 'COURSE' column.")
            st.stop()

        df['COURSE'] = df['COURSE'].astype(str).str.strip()
        feedback_cols = df.columns[METADATA_END_INDEX + 1:].tolist()
        
        if not feedback_cols:
            st.error("❌ No feedback columns found after metadata section.")
            st.stop()

        create_output_dir()

        # Enhanced Sidebar Controls
        with st.sidebar:
            st.markdown("## 🎨 Chart Customization")
            
            # Distribution Chart Settings
            st.markdown("### 📊 Distribution Chart")
            dist_chart_type = st.selectbox("Chart Type", ["Bar Chart", "Histogram", "Plotly Interactive"], key="dist_type")
            dist_color_palette = st.selectbox("Color Palette", 
                ["Modern Blues", "Sunset", "Professional", "Vibrant", "Purple Haze", "Ocean"], 
                key="dist_palette")
            
            if dist_chart_type != "Plotly Interactive":
                dist_width = st.slider("Width", 8, 20, 14, key="dist_width")
                dist_height = st.slider("Height", 6, 15, 8, key="dist_height")
                dist_bar_thickness = st.slider("Bar Thickness", 0.1, 2.0, 0.8, key="dist_thickness")
            else:
                dist_width, dist_height = 12, 8
                dist_bar_thickness = 0.8
                
            dist_font_size = st.slider("Font Size", 8, 16, 10, key="dist_font")
            dist_title_font = st.slider("Title Font", 14, 24, 18, key="dist_title_font")
            dist_x_rotation = st.slider("X-axis Label Rotation", 0, 90, 45, key="dist_rotation")
            dist_y_max = st.slider("Y-axis Maximum", 50, 150, 100, key="dist_y_max")
            dist_show_legend = st.checkbox("Show Legend", True, key="dist_legend")
            
            dist_title = st.text_input("Chart Title", "Response Distribution Analysis", key="dist_title")
            dist_xlabel = st.text_input("X-axis Label", "Feedback Questions", key="dist_xlabel")
            dist_ylabel = st.text_input("Y-axis Label", "Response Percentage (%)", key="dist_ylabel")

            st.markdown("---")
            
            # Average Scores Chart Settings
            st.markdown("### 📈 Average Scores Chart")
            avg_chart_type = st.selectbox("Chart Type", ["Bar Chart", "Plotly Interactive"], key="avg_type")
            avg_color_palette = st.selectbox("Color Palette", 
                ["Modern Blues", "Sunset", "Professional", "Vibrant", "Purple Haze", "Ocean"], 
                key="avg_palette")
            
            if avg_chart_type != "Plotly Interactive":
                avg_width = st.slider("Width", 8, 20, 12, key="avg_width")
                avg_height = st.slider("Height", 6, 15, 10, key="avg_height")
                avg_bar_thickness = st.slider("Bar Thickness", 0.5, 3.0, 1.0, key="avg_thickness")
            else:
                avg_width, avg_height = 12, 10
                avg_bar_thickness = 1.0
                
            avg_font_size = st.slider("Font Size", 8, 16, 10, key="avg_font")
            avg_title_font = st.slider("Title Font", 14, 24, 16, key="avg_title_font")
            avg_show_legend = st.checkbox("Show Legend", False, key="avg_legend")
            
            avg_title = st.text_input("Chart Title", "Average Scores by Question", key="avg_title")
            avg_xlabel = st.text_input("X-axis Label", "Average Score (1-5)", key="avg_xlabel")
            avg_ylabel = st.text_input("Y-axis Label", "Questions", key="avg_ylabel")

            st.markdown("---")
            
            # Pie Chart Settings
            st.markdown("### 🎯 Performance Overview")
            pie_chart_type = st.selectbox("Chart Type", ["Donut Chart", "Plotly Interactive"], key="pie_type")
            
            if pie_chart_type != "Plotly Interactive":
                pie_width = st.slider("Width", 4, 10, 6, key="pie_width")
                pie_height = st.slider("Height", 4, 10, 6, key="pie_height")
            else:
                pie_width, pie_height = 6, 6
                
            pie_donut_width = st.slider("Donut Width", 0.1, 0.7, 0.4, key="pie_donut")
            pie_title_font = st.slider("Title Font", 12, 24, 16, key="pie_title_font")
            pie_pct_font = st.slider("Percentage Font", 12, 28, 20, key="pie_pct_font")
            pie_show_pct = st.checkbox("Show Percentage", True, key="pie_show_pct")
            
            pie_title = st.text_input("Chart Title", "Overall Performance Score", key="pie_title")
            pie_color_main = st.color_picker("Primary Color", "#667eea", key="pie_main_color")
            pie_color_bg = st.color_picker("Background Color", "#e0e0e0", key="pie_bg_color")

        # Compile UI parameters
        ui_params = {
            "dist": {
                "chart_type": dist_chart_type, "color_palette": dist_color_palette,
                "width": dist_width, "height": dist_height, "bar_thickness": dist_bar_thickness,
                "font_size": dist_font_size, "title_font": dist_title_font, 
                "x_rotation": dist_x_rotation, "y_max": dist_y_max, "show_legend": dist_show_legend,
                "title": dist_title, "x_label": dist_xlabel, "y_label": dist_ylabel
            },
            "avg": {
                "chart_type": avg_chart_type, "color_palette": avg_color_palette,
                "width": avg_width, "height": avg_height, "bar_thickness": avg_bar_thickness,
                "font_size": avg_font_size, "title_font": avg_title_font, "show_legend": avg_show_legend,
                "title": avg_title, "x_label": avg_xlabel, "y_label": avg_ylabel
            },
            "pie": {
                "chart_type": pie_chart_type, "width": pie_width, "height": pie_height,
                "donut_width": pie_donut_width, "title_font": pie_title_font, 
                "pct_font": pie_pct_font, "show_pct": pie_show_pct, "title": pie_title,
                "color_main": pie_color_main, "color_bg": pie_color_bg
            }
        }

        # Course selection
        courses = sorted(df['COURSE'].unique())
        selected_course = st.selectbox(
            "🎓 Select Course for Analysis", 
            courses,
            help="Choose a course to display detailed analytics"
        )

        # Process and display charts
        if selected_course:
            process_course_enhanced(df, selected_course, feedback_cols, ui_params)

            # Enhanced download functionality
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📥 Download Distribution Chart", use_container_width=True):
                    st.info("Chart download feature will be implemented in the production version!")
            
            with col2:
                if st.button("📥 Download Average Scores", use_container_width=True):
                    st.info("Chart download feature will be implemented in the production version!")
            
            with col3:
                if st.button("📥 Download All Charts", use_container_width=True):
                    st.info("Bulk download feature will be implemented in the production version!")

        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem;">
            <p>📊 Teacher Feedback Analytics Platform | Built with Streamlit & Python</p>
            <p style="font-size: 0.8em;">© 2024 - Premium Data Visualization Suite</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
