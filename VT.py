# interactive_feedback_app.py
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
def plot_distribution(df_long, course, fig_w, fig_h, title_font, label_font, tick_font,
                      bar_palette, x_rotation, y_max, show_legend, custom_title, x_label, y_label):
    count_df = df_long.groupby(['Question','Response']).size().reset_index(name='Count')
    total_per_question = count_df.groupby('Question')['Count'].transform('sum')
    count_df['Percentage'] = count_df['Count'] / total_per_question * 100
    questions = list(pd.Categorical(count_df['Question']).categories)
    count_df['Question'] = pd.Categorical(count_df['Question'], categories=questions, ordered=True)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.barplot(
        data=count_df,
        x='Question',
        y='Percentage',
        hue='Response',
        palette=bar_palette,
        ax=ax,
        order=questions
    )

    wrapped_labels = ["\n".join(textwrap.wrap(str(q), 25)) for q in questions]
    ax.set_xticklabels(wrapped_labels, rotation=x_rotation, ha='center', fontsize=tick_font)
    ax.set_xlabel(x_label, fontsize=label_font)
    ax.set_ylabel(y_label, fontsize=label_font)
    ax.set_title(custom_title, fontsize=title_font, pad=16)
    ax.set_ylim(0, y_max)

    if show_legend:
        ax.legend(title='Response (1-5)', bbox_to_anchor=(1.02, 0.5), loc='center left')
    else:
        ax.get_legend().remove()

    st.pyplot(fig)
    return fig

def plot_average_scores(mean_scores, course, fig_w, fig_h, title_font, label_font, tick_font,
                        bar_palette, x_label, y_label, show_legend, custom_title):
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.barplot(
        y=mean_scores.index,
        x=mean_scores.values,
        palette=bar_palette,
        ax=ax
    )
    ax.set_xlim(1, 5)
    ax.set_xlabel(x_label, fontsize=label_font)
    ax.set_ylabel(y_label, fontsize=label_font)
    ax.set_title(custom_title, fontsize=title_font)
    ax.tick_params(axis='y', labelsize=tick_font)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=4)
    if not show_legend:
        ax.get_legend().remove()
    st.pyplot(fig)
    return fig

def plot_cumulative_pie(course, percent, fig_w, fig_h, donut_width, title_font, pct_font,
                        show_percentage, custom_title, color_main, color_bg):
    score = max(min(percent, 100), 0)
    remainder = 100 - score
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    wedges, texts = ax.pie(
        [score, remainder],
        labels=['', ''],
        colors=[color_main, color_bg],
        startangle=90,
        wedgeprops={'width': donut_width, 'edgecolor': 'white'}
    )

    if show_percentage:
        ax.text(0, 0.05, f'{score:.1f}%', ha='center', va='center',
                fontsize=pct_font, fontweight='bold', color='black')
        ax.text(0, -0.18, 'Mean Score', ha='center', va='center', fontsize=pct_font-6, color='black')

    ax.set_title(custom_title, fontsize=title_font, pad=14)
    for t in texts:
        t.set_text('')
    st.pyplot(fig)
    return fig

# --- Course processing ---
def process_course(df, course, feedback_cols, ui_params):
    st.subheader(f"Course: {course}")
    df_course = df[df['COURSE'] == course].copy()
    if df_course.empty:
        st.warning(f"No data for {course}")
        return

    current_cols = [col for col in feedback_cols if col in df_course.columns]
    if not current_cols:
        st.warning(f"No feedback columns for {course}")
        return

    df_numeric = df_course[current_cols].apply(pd.to_numeric, errors='coerce')
    df_long = df_numeric.melt(var_name='Question', value_name='Response').dropna()

    if not df_long.empty:
        # Distribution chart
        plot_distribution(
            df_long, course,
            fig_w=ui_params['dist_fig_w'], fig_h=ui_params['dist_fig_h'],
            title_font=ui_params['dist_title_font'], label_font=ui_params['dist_label_font'],
            tick_font=ui_params['dist_tick_font'], bar_palette=ui_params['dist_palette'],
            x_rotation=ui_params['dist_x_rotation'], y_max=ui_params['dist_y_max'],
            show_legend=ui_params['dist_show_legend'],
            custom_title=ui_params['dist_title'], x_label=ui_params['dist_xlabel'],
            y_label=ui_params['dist_ylabel']
        )

        # Average scores chart
        mean_scores = df_numeric.mean().sort_values()
        plot_average_scores(
            mean_scores, course,
            fig_w=ui_params['avg_fig_w'], fig_h=ui_params['avg_fig_h'],
            title_font=ui_params['avg_title_font'], label_font=ui_params['avg_label_font'],
            tick_font=ui_params['avg_tick_font'], bar_palette=ui_params['avg_palette'],
            x_label=ui_params['avg_xlabel'], y_label=ui_params['avg_ylabel'],
            show_legend=ui_params['avg_show_legend'],
            custom_title=ui_params['avg_title']
        )

        # Cumulative pie
        flat = df_numeric.values.flatten()
        pct = calculate_cumulative_percentage(pd.Series(flat))
        st.info(f"Cumulative Mean Percentage: {pct:.2f}%")
        plot_cumulative_pie(
            course, pct,
            fig_w=ui_params['pie_fig_w'], fig_h=ui_params['pie_fig_h'],
            donut_width=ui_params['pie_donut_width'],
            title_font=ui_params['pie_title_font'], pct_font=ui_params['pie_pct_font'],
            show_percentage=ui_params['pie_show_pct'],
            custom_title=ui_params['pie_title'],
            color_main=ui_params['pie_color_main'], color_bg=ui_params['pie_color_bg']
        )
    else:
        st.warning(f"No valid numeric responses for {course}")

# --- Streamlit UI ---
st.title("Data Visualization Tool")
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

    # --- Sidebar UI controls ---
    st.sidebar.header("Customize Distribution Chart")
    dist_fig_w = st.sidebar.slider("Width", 5, 20, 14)
    dist_fig_h = st.sidebar.slider("Height", 4, 15, 8)
    dist_title_font = st.sidebar.slider("Title font", 10, 30, 18)
    dist_label_font = st.sidebar.slider("Axis label font", 8, 20, 12)
    dist_tick_font = st.sidebar.slider("Tick label font", 6, 16, 10)
    dist_palette = st.sidebar.selectbox("Color palette", ["viridis","magma","plasma","coolwarm"])
    dist_x_rotation = st.sidebar.slider("X-axis rotation", 0, 90, 90)
    dist_y_max = st.sidebar.slider("Y-axis max", 50, 150, 100)
    dist_show_legend = st.sidebar.checkbox("Show legend", True)
    dist_title = st.sidebar.text_input("Custom title", "Response Distribution (% per Question)")
    dist_xlabel = st.sidebar.text_input("X-axis label", "Feedback Question")
    dist_ylabel = st.sidebar.text_input("Y-axis label", "Percentage of Responses (%)")

    st.sidebar.header("Customize Average Scores Chart")
    avg_fig_w = st.sidebar.slider("Width (avg)", 5, 20, 12)
    avg_fig_h = st.sidebar.slider("Height (avg)", 4, 15, 10)
    avg_title_font = st.sidebar.slider("Title font (avg)", 10, 30, 16)
    avg_label_font = st.sidebar.slider("Axis label font (avg)", 8, 20, 12)
    avg_tick_font = st.sidebar.slider("Tick font (avg)", 6, 16, 10)
    avg_palette = st.sidebar.selectbox("Color palette (avg)", ["viridis","magma","plasma","coolwarm"], index=0)
    avg_show_legend = st.sidebar.checkbox("Show legend (avg)", True)
    avg_title = st.sidebar.text_input("Custom title (avg)", "Average Scores")
    avg_xlabel = st.sidebar.text_input("X-axis label (avg)", "Average Score (1–5)")
    avg_ylabel = st.sidebar.text_input("Y-axis label (avg)", "Feedback Question")

    st.sidebar.header("Customize Cumulative Pie Chart")
    pie_fig_w = st.sidebar.slider("Width (pie)", 4, 10, 6)
    pie_fig_h = st.sidebar.slider("Height (pie)", 4, 10, 6)
    pie_donut_width = st.sidebar.slider("Donut width", 0.1, 0.9, 0.4)
    pie_title_font = st.sidebar.slider("Title font (pie)", 10, 30, 16)
    pie_pct_font = st.sidebar.slider("Percentage font (pie)", 8, 24, 18)
    pie_show_pct = st.sidebar.checkbox("Show percentage", True)
    pie_title = st.sidebar.text_input("Custom title (pie)", "Cumulative Mean Score")
    pie_color_main = st.sidebar.color_picker("Main color", "#43a047")
    pie_color_bg = st.sidebar.color_picker("Background color", "#e0e0e0")

    ui_params = {
        "dist_fig_w": dist_fig_w, "dist_fig_h": dist_fig_h, "dist_title_font": dist_title_font,
        "dist_label_font": dist_label_font, "dist_tick_font": dist_tick_font, "dist_palette": dist_palette,
        "dist_x_rotation": dist_x_rotation, "dist_y_max": dist_y_max, "dist_show_legend": dist_show_legend,
        "dist_title": dist_title, "dist_xlabel": dist_xlabel, "dist_ylabel": dist_ylabel,
        "avg_fig_w": avg_fig_w, "avg_fig_h": avg_fig_h, "avg_title_font": avg_title_font,
        "avg_label_font": avg_label_font, "avg_tick_font": avg_tick_font, "avg_palette": avg_palette,
        "avg_show_legend": avg_show_legend, "avg_title": avg_title, "avg_xlabel": avg_xlabel, "avg_ylabel": avg_ylabel,
        "pie_fig_w": pie_fig_w, "pie_fig_h": pie_fig_h, "pie_donut_width": pie_donut_width,
        "pie_title_font": pie_title_font, "pie_pct_font": pie_pct_font, "pie_show_pct": pie_show_pct,
        "pie_title": pie_title, "pie_color_main": pie_color_main, "pie_color_bg": pie_color_bg
    }

    selected_course = st.selectbox("Select Course to Display", df['COURSE'].unique())
    process_course(df, selected_course, feedback_cols, ui_params)

# --- Footer ---
st.markdown("---")
st.markdown("Developed by **Department of Computer Science, Central University of Tamilnadu**. All rights reserved.")
