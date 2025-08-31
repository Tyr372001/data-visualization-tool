# interactive_feedback_app_v3.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import os
import re

# --- Configuration ---
METADATA_END_INDEX = 8
OUTPUT_DIR = "Output_Visuals"

# --- Utilities ---
def create_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def sanitize_filename(name):
    return re.sub(r'[\\/:"*?<>|]+', '_', name).strip()

def calculate_cumulative_percentage(series):
    valid = series.dropna()
    score = valid.mean() if len(valid) > 0 else 0
    return (score / 5) * 100  # assuming 5 is max score

# --- Plotting functions ---
def plot_distribution(df_long, ui_params):
    count_df = df_long.groupby(['Question','Response']).size().reset_index(name='Count')
    total_per_question = count_df.groupby('Question')['Count'].transform('sum')
    count_df['Percentage'] = count_df['Count'] / total_per_question * 100
    questions = list(pd.Categorical(count_df['Question']).categories)
    count_df['Question'] = pd.Categorical(count_df['Question'], categories=questions, ordered=True)

    fig, ax = plt.subplots(figsize=(ui_params['dist_fig_w'], ui_params['dist_fig_h']))

    if ui_params['dist_chart_type'] == "Bar Plot":
        sns.barplot(
            data=count_df, x='Question', y='Percentage', hue='Response',
            palette=ui_params['dist_palette'], ax=ax, order=questions, width=ui_params['dist_bar_width']
        )
    else:  # Histogram
        for q in questions:
            vals = df_long[df_long['Question'] == q]['Response']
            ax.hist(vals, bins=5, alpha=0.6, label="\n".join(textwrap.wrap(q,25)))

    wrapped_labels = ["\n".join(textwrap.wrap(str(q), 25)) for q in questions]
    ax.set_xticklabels(wrapped_labels, rotation=ui_params['dist_x_rotation'], ha='center', fontsize=ui_params['dist_tick_font'])
    ax.set_xlabel(ui_params['dist_xlabel'], fontsize=ui_params['dist_label_font'])
    ax.set_ylabel(ui_params['dist_ylabel'], fontsize=ui_params['dist_label_font'])
    ax.set_title(ui_params['dist_title'], fontsize=ui_params['dist_title_font'], pad=16)
    ax.set_ylim(0, ui_params['dist_y_max'])

    if ui_params['dist_show_legend']:
        ax.legend(title='Response (1-5)', bbox_to_anchor=(1.02, 0.5), loc='center left')
    else:
        ax.get_legend().remove()

    st.pyplot(fig)
    return fig

def plot_average_scores(mean_scores, ui_params):
    fig, ax = plt.subplots(figsize=(ui_params['avg_fig_w'], ui_params['avg_fig_h']))

    if ui_params['avg_chart_type'] == "Horizontal Bar":
        sns.barplot(
            y=mean_scores.index, x=mean_scores.values, palette=ui_params['avg_palette'],
            ax=ax, orient='h', width=ui_params['avg_bar_width']
        )
        ax.set_xlim(1,5)
    else:  # Vertical Histogram
        ax.hist(mean_scores.values, bins=5, alpha=0.7, color='skyblue')
        ax.set_xlim(1,5)

    ax.set_xlabel(ui_params['avg_xlabel'], fontsize=ui_params['avg_label_font'])
    ax.set_ylabel(ui_params['avg_ylabel'], fontsize=ui_params['avg_label_font'])
    ax.set_title(ui_params['avg_title'], fontsize=ui_params['avg_title_font'])
    ax.tick_params(axis='y', labelsize=ui_params['avg_tick_font'])

    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=4)

    if not ui_params['avg_show_legend']:
        ax.get_legend().remove()

    st.pyplot(fig)
    return fig

def plot_cumulative_pie(course, percent, ui_params):
    score = max(min(percent, 100), 0)
    remainder = 100 - score
    fig, ax = plt.subplots(figsize=(ui_params['pie_fig_w'], ui_params['pie_fig_h']))

    width = ui_params['pie_donut_width'] if ui_params['pie_type']=="Donut" else 1.0
    wedges, texts = ax.pie(
        [score, remainder],
        labels=['',''],
        colors=[ui_params['pie_color_main'], ui_params['pie_color_bg']],
        startangle=90,
        wedgeprops={'width': width, 'edgecolor':'white'}
    )

    if ui_params['pie_show_pct']:
        ax.text(0,0.05,f'{score:.1f}%', ha='center', va='center',
                fontsize=ui_params['pie_pct_font'], fontweight='bold', color='black')
        ax.text(0,-0.18,'Mean Score', ha='center', va='center', fontsize=ui_params['pie_pct_font']-6, color='black')

    ax.set_title(ui_params['pie_title'], fontsize=ui_params['pie_title_font'], pad=14)
    for t in texts:
        t.set_text('')
    st.pyplot(fig)
    return fig

# --- Course processing ---
def process_course(df, course, feedback_cols, ui_params):
    st.subheader(f"Course: {course}")
    df_course = df[df['COURSE']==course].copy()
    if df_course.empty:
        st.warning(f"No data for {course}")
        return

    current_cols = [col for col in feedback_cols if col in df_course.columns]
    if not current_cols:
        st.warning(f"No feedback columns for {course}")
        return

    df_numeric = df_course[current_cols].apply(pd.to_numeric, errors='coerce')
    df_long = df_numeric.melt(var_name='Question', value_name='Response').dropna()

    # KPI Cards
    st.markdown("### 📈 Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Score", f"{df_numeric.mean().mean():.2f}")
    col2.metric("Median Score", f"{df_numeric.median().median():.2f}")
    col3.metric("Total Responses", f"{df_long.shape[0]}")

    if not df_long.empty:
        plot_distribution(df_long, ui_params)
        mean_scores = df_numeric.mean().sort_values()
        plot_average_scores(mean_scores, ui_params)
        pct = calculate_cumulative_percentage(pd.Series(df_numeric.values.flatten()))
        st.info(f"Cumulative Mean Percentage: {pct:.2f}%")
        plot_cumulative_pie(course, pct, ui_params)
    else:
        st.warning(f"No valid numeric responses for {course}")

# --- Streamlit UI ---
st.title("📊 Interactive Data Visualization Tool")
st.write("Upload CSV and customize charts live!")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.stop()

    if 'COURSE' not in df.columns:
        st.error("CSV must contain a 'COURSE' column.")
        st.stop()

    df['COURSE'] = df['COURSE'].astype(str).str.strip()
    feedback_cols = df.columns[METADATA_END_INDEX + 1:].tolist()
    if not feedback_cols:
        st.error("No feedback columns found.")
        st.stop()

    create_output_dir()

    # --- Sidebar with grouped expanders ---
    st.sidebar.header("⚙️ Customization Panel")

    with st.sidebar.expander("📊 Distribution Chart", expanded=True):
        dist_chart_type = st.radio("Chart Type", ["Bar Plot","Histogram"], index=0, key="dist_chart_type")
        dist_bar_width = st.slider("Bar Thickness", 0.1,1.0,0.8, key="dist_bar_width")
        dist_fig_w = st.slider("Width", 5,20,14, key="dist_fig_w")
        dist_fig_h = st.slider("Height",4,15,8, key="dist_fig_h")
        dist_title_font = st.slider("Title Font Size", 10,30,18, key="dist_title_font")
        dist_label_font = st.slider("Axis Label Font Size",8,20,12, key="dist_label_font")
        dist_tick_font = st.slider("Tick Label Font Size",6,16,10, key="dist_tick_font")
        dist_palette = st.selectbox("Color Palette", ["viridis","magma","plasma","coolwarm"], key="dist_palette")
        dist_x_rotation = st.slider("X-axis Rotation",0,90,90, key="dist_x_rotation")
        dist_y_max = st.slider("Y-axis Max",50,150,100, key="dist_y_max")
        dist_show_legend = st.checkbox("Show Legend", True, key="dist_show_legend")
        dist_title = st.text_input("Custom Title", "Response Distribution (% per Question)", key="dist_title")
        dist_xlabel = st.text_input("X-axis Label", "Feedback Question", key="dist_xlabel")
        dist_ylabel = st.text_input("Y-axis Label", "Percentage of Responses (%)", key="dist_ylabel")

    with st.sidebar.expander("📊 Average Scores Chart", expanded=False):
        avg_chart_type = st.radio("Chart Type", ["Horizontal Bar","Vertical Histogram"], index=0, key="avg_chart_type")
        avg_bar_width = st.slider("Bar Thickness",0.1,1.0,0.8, key="avg_bar_width")
        avg_fig_w = st.slider("Width",5,20,12, key="avg_fig_w")
        avg_fig_h = st.slider("Height",4,15,10, key="avg_fig_h")
        avg_title_font = st.slider("Title Font Size",10,30,16, key="avg_title_font")
        avg_label_font = st.slider("Axis Label Font Size",8,20,12, key="avg_label_font")
        avg_tick_font = st.slider("Tick Label Font Size",6,16,10, key="avg_tick_font")
        avg_palette = st.selectbox("Color Palette", ["viridis","magma","plasma","coolwarm"], index=0, key="avg_palette")
        avg_show_legend = st.checkbox("Show Legend", True, key="avg_show_legend")
        avg_title = st.text_input("Custom Title", "Average Scores", key="avg_title")
        avg_xlabel = st.text_input("X-axis Label", "Average Score (1–5)", key="avg_xlabel")
        avg_ylabel = st.text_input("Y-axis Label", "Feedback Question", key="avg_ylabel")

    with st.sidebar.expander("🥧 Cumulative Pie/Donut Chart", expanded=False):
        pie_type = st.radio("Chart Type", ["Donut","Pie"], index=0, key="pie_type")
        pie_donut_width = st.slider("Donut Width",0.1,0.9,0.4, key="pie_donut_width")
        pie_fig_w = st.slider("Width",4,10,6, key="pie_fig_w")
        pie_fig_h = st.slider("Height",4,10,6, key="pie_fig_h")
        pie_title_font = st.slider("Title Font Size",10,30,16, key="pie_title_font")
        pie_pct_font = st.slider("Percentage Font Size",8,24,18, key="pie_pct_font")
        pie_show_pct = st.checkbox("Show Percentage", True, key="pie_show_pct")
        pie_title = st.text_input("Custom Title", "Cumulative Mean Score", key="pie_title")
        pie_color_main = st.color_picker("Main Color", "#43a047", key="pie_color_main")
        pie_color_bg = st.color_picker("Background Color", "#e0e0e0", key="pie_color_bg")

    # --- Collect UI parameters ---
    ui_params = {
        "dist_chart_type": dist_chart_type, "dist_bar_width": dist_bar_width, "dist_fig_w": dist_fig_w, "dist_fig_h": dist_fig_h,
        "dist_title_font": dist_title_font, "dist_label_font": dist_label_font, "dist_tick_font": dist_tick_font,
        "dist_palette": dist_palette, "dist_x_rotation": dist_x_rotation, "dist_y_max": dist_y_max,
        "dist_show_legend": dist_show_legend, "dist_title": dist_title, "dist_xlabel": dist_xlabel, "dist_ylabel": dist_ylabel,
        "avg_chart_type": avg_chart_type, "avg_bar_width": avg_bar_width, "avg_fig_w": avg_fig_w, "avg_fig_h": avg_fig_h,
        "avg_title_font": avg_title_font, "avg_label_font": avg_label_font, "avg_tick_font": avg_tick_font, "avg_palette": avg_palette,
        "avg_show_legend": avg_show_legend, "avg_title": avg_title, "avg_xlabel": avg_xlabel, "avg_ylabel": avg_ylabel,
        "pie_type": pie_type, "pie_donut_width": pie_donut_width, "pie_fig_w": pie_fig_w, "pie_fig_h": pie_fig_h,
        "pie_title_font": pie_title_font, "pie_pct_font": pie_pct_font, "pie_show_pct": pie_show_pct,
        "pie_title": pie_title, "pie_color_main": pie_color_main, "pie_color_bg": pie_color_bg
    }

    selected_course = st.selectbox("Select Course to Display", df['COURSE'].unique(), key="selected_course")
    process_course(df, selected_course, feedback_cols, ui_params)

# --- Footer ---
st.markdown("---")
st.markdown("Developed by **Department of Computer Science, Central University of Tamilnadu. Created By Subhradeep Sarkar P241321 & Dr. Thiyagarajan P**. All rights reserved.")
