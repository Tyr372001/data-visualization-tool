# interactive_visualization_tool.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from io import BytesIO
import re
import numpy as np

# --- Utilities ---
def sanitize_filename(name):
    return re.sub(r'[\\/:"*?<>|]+', '_', name).strip()

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

# --- Header with Logos ---
col1, col2 = st.columns([1, 3])
try:
    app_logo = Image.open("app_logo.png")
    uni_logo = Image.open("university_logo.png")
    col1.image(app_logo, width=100)
    col2.image(uni_logo, width=150)
except Exception as e:
    st.warning(f"Logos not found: {e}")

st.title("Data Visualization Tool")
st.write("Upload any CSV and interactively visualize your data.")

# --- File upload ---
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.stop()

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(exclude='number').columns.tolist()

    # Sidebar controls
    st.sidebar.header("Column Selection & Formula")
    group_col = st.sidebar.selectbox("Categorical Column (for grouping)", [None]+categorical_cols)
    value_col = st.sidebar.selectbox("Numeric Column (for value)", numeric_cols)
    custom_formula = st.sidebar.text_input("Custom Formula (optional, e.g., (ColA + ColB)/2)")

    if custom_formula:
        try:
            df["CustomCol"] = pd.eval(custom_formula, engine='python', local_dict=df.to_dict('series'))
            value_col = "CustomCol"
            st.success("Custom formula applied successfully.")
        except Exception as e:
            st.error(f"Error in formula: {e}")

    st.sidebar.header("Chart Settings")
    chart_type = st.sidebar.selectbox("Select Chart Type", ["Bar","Boxplot","Pie","Line","Scatter","Area","Violin","Histogram","Pairplot"])
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

    # --- Plotting Functions ---
    def plot_bar(df, group_col, value_col):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.barplot(data=df, x=group_col, y=value_col, palette=palette, ax=ax)
        ax.set_title(chart_title, fontsize=title_font)
        ax.set_xlabel(group_col, fontsize=label_font)
        ax.set_ylabel(value_col, fontsize=label_font)
        ax.tick_params(axis='x', rotation=x_rotation, labelsize=tick_font)
        st.pyplot(fig)
        add_download_button(fig, chart_title)

    def plot_box(df, group_col, value_col):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.boxplot(data=df, x=group_col, y=value_col, palette=palette, ax=ax)
        ax.set_title(chart_title, fontsize=title_font)
        ax.set_xlabel(group_col, fontsize=label_font)
        ax.set_ylabel(value_col, fontsize=label_font)
        ax.tick_params(axis='x', rotation=x_rotation, labelsize=tick_font)
        st.pyplot(fig)
        add_download_button(fig, chart_title)

    def plot_pie(df, group_col, value_col):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
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
                ax.text(x*0.6, y*0.6, f'{scores[i]:.1f}', ha='center', va='center', fontsize=tick_font)
        ax.set_title(chart_title, fontsize=title_font)
        st.pyplot(fig)
        add_download_button(fig, chart_title)

    def plot_line(df, x_col, y_col):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.plot(df[x_col], df[y_col], marker='o', color=sns.color_palette(palette)[0])
        ax.set_title(chart_title, fontsize=title_font)
        ax.set_xlabel(x_col, fontsize=label_font)
        ax.set_ylabel(y_col, fontsize=label_font)
        ax.tick_params(axis='x', rotation=x_rotation, labelsize=tick_font)
        st.pyplot(fig)
        add_download_button(fig, chart_title)

    def plot_scatter(df, x_col, y_col):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=group_col, palette=palette, ax=ax)
        ax.set_title(chart_title, fontsize=title_font)
        ax.set_xlabel(x_col, fontsize=label_font)
        ax.set_ylabel(y_col, fontsize=label_font)
        ax.tick_params(axis='x', rotation=x_rotation, labelsize=tick_font)
        st.pyplot(fig)
        add_download_button(fig, chart_title)

    def plot_area(df, x_col, y_col):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.fill_between(df[x_col], df[y_col], color=sns.color_palette(palette)[0], alpha=0.5)
        ax.plot(df[x_col], df[y_col], color=sns.color_palette(palette)[0])
        ax.set_title(chart_title, fontsize=title_font)
        ax.set_xlabel(x_col, fontsize=label_font)
        ax.set_ylabel(y_col, fontsize=label_font)
        ax.tick_params(axis='x', rotation=x_rotation, labelsize=tick_font)
        st.pyplot(fig)
        add_download_button(fig, chart_title)

    def plot_violin(df, x_col, y_col):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.violinplot(data=df, x=x_col, y=y_col, palette=palette, ax=ax)
        ax.set_title(chart_title, fontsize=title_font)
        ax.set_xlabel(x_col, fontsize=label_font)
        ax.set_ylabel(y_col, fontsize=label_font)
        ax.tick_params(axis='x', rotation=x_rotation, labelsize=tick_font)
        st.pyplot(fig)
        add_download_button(fig, chart_title)

    def plot_hist(df, value_col):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.histplot(df[value_col], bins=bins, kde=True, color=sns.color_palette(palette)[0], ax=ax)
        ax.set_title(chart_title, fontsize=title_font)
        ax.set_xlabel(value_col, fontsize=label_font)
        ax.set_ylabel("Count", fontsize=label_font)
        ax.tick_params(axis='x', rotation=x_rotation, labelsize=tick_font)
        st.pyplot(fig)
        add_download_button(fig, chart_title)

    def plot_pairplot(df):
        fig = sns.pairplot(df)
        st.pyplot(fig)
        st.info("Download Pairplot manually if needed (save as image).")

    # --- Render selected chart ---
    if chart_type == "Bar" and group_col and value_col:
        plot_bar(df, group_col, value_col)
    elif chart_type == "Boxplot" and group_col and value_col:
        plot_box(df, group_col, value_col)
    elif chart_type == "Pie" and group_col and value_col:
        plot_pie(df, group_col, value_col)
    elif chart_type == "Line" and group_col and value_col:
        plot_line(df, group_col, value_col)
    elif chart_type == "Scatter" and group_col and value_col:
        plot_scatter(df, group_col, value_col)
    elif chart_type == "Area" and group_col and value_col:
        plot_area(df, group_col, value_col)
    elif chart_type == "Violin" and group_col and value_col:
        plot_violin(df, group_col, value_col)
    elif chart_type == "Histogram" and value_col:
        plot_hist(df, value_col)
    elif chart_type == "Pairplot" and len(df.select_dtypes(include='number').columns) > 1:
        plot_pairplot(df.select_dtypes(include='number'))
    else:
        st.warning("Please select appropriate columns for the chosen chart type.")

# --- Footer ---
st.markdown("---")
st.markdown("Developed by **Department of Computer Science, Central University of Tamilnadu**. All rights reserved.")
