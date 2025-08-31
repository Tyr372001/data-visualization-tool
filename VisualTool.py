# interactive_feedback_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import re
import os
import numpy as np
from io import BytesIO

# --- Configuration ---
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
    return score

def add_download_button(fig, title):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    st.download_button(
        label=f"Download '{title}' as PNG",
        data=buf,
        file_name=f"{sanitize_filename(title)}.png",
        mime="image/png"
    )

# --- Plotting Functions ---
def plot_bar(df, group_col, value_col, title, xlabel, ylabel, palette, fig_w, fig_h, title_font, label_font, tick_font, x_rotation):
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.barplot(data=df, x=group_col, y=value_col, palette=palette, ax=ax)
    ax.set_title(title, fontsize=title_font)
    ax.set_xlabel(xlabel, fontsize=label_font)
    ax.set_ylabel(ylabel, fontsize=label_font)
    ax.tick_params(axis='x', rotation=x_rotation, labelsize=tick_font)
    st.pyplot(fig)
    add_download_button(fig, title)
    return fig

def plot_box(df, group_col, value_col, title, xlabel, ylabel, palette, fig_w, fig_h, title_font, label_font, tick_font, x_rotation):
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.boxplot(data=df, x=group_col, y=value_col, palette=palette, ax=ax)
    ax.set_title(title, fontsize=title_font)
    ax.set_xlabel(xlabel, fontsize=label_font)
    ax.set_ylabel(ylabel, fontsize=label_font)
    ax.tick_params(axis='x', rotation=x_rotation, labelsize=tick_font)
    st.pyplot(fig)
    add_download_button(fig, title)
    return fig

def plot_hist(df, value_col, title, xlabel, ylabel, palette, fig_w, fig_h, title_font, label_font, tick_font, bins):
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.histplot(df[value_col], kde=True, color=palette, bins=bins, ax=ax)
    ax.set_title(title, fontsize=title_font)
    ax.set_xlabel(xlabel, fontsize=label_font)
    ax.set_ylabel(ylabel, fontsize=label_font)
    ax.tick_params(axis='x', labelsize=tick_font)
    st.pyplot(fig)
    add_download_button(fig, title)
    return fig

def plot_pie(df, group_col, value_col, title, donut_width, title_font, pct_font, show_pct):
    fig, ax = plt.subplots(figsize=(6,6))
    data = df.groupby(group_col)[value_col].mean().reset_index()
    scores = data[value_col].tolist()
    labels = data[group_col].tolist()
    wedges, texts = ax.pie(scores, labels=labels, startangle=90, colors=sns.color_palette("pastel", len(labels)),
                           wedgeprops={'width': donut_width, 'edgecolor': 'white'})
    if show_pct:
        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            ax.text(x*0.6, y*0.6, f'{scores[i]:.1f}', ha='center', va='center', fontsize=pct_font)
    ax.set_title(title, fontsize=title_font)
    st.pyplot(fig)
    add_download_button(fig, title)
    return fig

# --- Streamlit App ---
st.set_page_config(page_title="Data Visualization Tool", layout="wide")
st.title("Data Visualization Tool")
st.markdown("Upload any CSV and interactively visualize your data.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.stop()

    create_output_dir()

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(exclude='number').columns.tolist()

    st.sidebar.header("Column Selection")
    group_col = st.sidebar.selectbox("Select Categorical Column (for grouping)", [None]+categorical_cols)
    value_col = st.sidebar.selectbox("Select Numeric Column (for value)", numeric_cols)

    st.sidebar.header("Custom Formula (optional)")
    custom_formula = st.sidebar.text_input("Enter formula using numeric columns (e.g., (ColA + ColB)/2)")
    if custom_formula:
        try:
            df["CustomCol"] = pd.eval(custom_formula, engine='python', local_dict=df.to_dict('series'))
            value_col = "CustomCol"
            st.success("Custom formula applied successfully.")
        except Exception as e:
            st.error(f"Error in formula: {e}")

    st.sidebar.header("Chart Type & Settings")
    chart_type = st.sidebar.selectbox("Select Chart Type", ["Bar", "Boxplot", "Histogram", "Pie"])

    fig_w = st.sidebar.slider("Figure width", 5, 20, 12)
    fig_h = st.sidebar.slider("Figure height", 4, 15, 8)
    title_font = st.sidebar.slider("Title font size", 10, 30, 16)
    label_font = st.sidebar.slider("Axis label font size", 8, 20, 12)
    tick_font = st.sidebar.slider("Tick font size", 6, 16, 10)
    x_rotation = st.sidebar.slider("X-axis label rotation", 0, 90, 45)
    palette = st.sidebar.selectbox("Color palette", ["viridis","magma","plasma","coolwarm","pastel"])
    bins = st.sidebar.slider("Histogram bins", 5, 50, 20)
    donut_width = st.sidebar.slider("Pie donut width", 0.1, 0.9, 0.4)
    show_pct = st.sidebar.checkbox("Show values on Pie chart", True)
    chart_title = st.sidebar.text_input("Chart Title", "Data Visualization")

    # Plot based on selection
    if chart_type == "Bar" and group_col and value_col:
        plot_bar(df, group_col, value_col, chart_title, group_col, value_col, palette, fig_w, fig_h, title_font, label_font, tick_font, x_rotation)
    elif chart_type == "Boxplot" and group_col and value_col:
        plot_box(df, group_col, value_col, chart_title, group_col, value_col, palette, fig_w, fig_h, title_font, label_font, tick_font, x_rotation)
    elif chart_type == "Histogram" and value_col:
        plot_hist(df, value_col, chart_title, value_col, "Count", palette, fig_w, fig_h, title_font, label_font, tick_font, bins)
    elif chart_type == "Pie" and group_col and value_col:
        plot_pie(df, group_col, value_col, chart_title, donut_width, title_font, tick_font, show_pct)
    else:
        st.warning("Please select appropriate columns for the chosen chart type.")

# Footer
st.markdown("---")
st.markdown("Developed by **Department of Computer Science, Central University of Tamilnadu**. All rights reserved.")
