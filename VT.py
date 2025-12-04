import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import os
import re
import io
import zipfile

# --- Configuration ---
METADATA_END_INDEX = 8
OUTPUT_DIR = "Teacher_Output"

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

# --- KPI Calculation Functions ---
def calculate_kpis(df_course, feedback_cols):
    """Calculate meaningful KPIs from the feedback data"""
    current_cols = [col for col in feedback_cols if col in df_course.columns]
    if not current_cols:
        return None
    
    df_numeric = df_course[current_cols].apply(pd.to_numeric, errors='coerce')
    
    # Total responses
    total_responses = len(df_course)
    
    # Average score (mean of all feedback)
    all_scores = df_numeric.values.flatten()
    valid_scores = all_scores[~pd.isna(all_scores)]
    avg_score = valid_scores.mean() if len(valid_scores) > 0 else 0
    
    # Satisfaction rate (% of 4-5 ratings)
    satisfaction_count = sum((valid_scores >= 4))
    satisfaction_rate = (satisfaction_count / len(valid_scores) * 100) if len(valid_scores) > 0 else 0
    
    # Response rate (assuming total possible is total responses * number of questions)
    total_possible = len(df_course) * len(current_cols)
    response_rate = (len(valid_scores) / total_possible * 100) if total_possible > 0 else 0
    
    # Lowest scoring question
    mean_by_question = df_numeric.mean()
    lowest_question = mean_by_question.idxmin() if not mean_by_question.empty else "N/A"
    lowest_score = mean_by_question.min() if not mean_by_question.empty else 0
    
    # Highest scoring question
    highest_question = mean_by_question.idxmax() if not mean_by_question.empty else "N/A"
    highest_score = mean_by_question.max() if not mean_by_question.empty else 0
    
    return {
        'total_responses': total_responses,
        'avg_score': avg_score,
        'satisfaction_rate': satisfaction_rate,
        'response_rate': response_rate,
        'lowest_question': lowest_question,
        'lowest_score': lowest_score,
        'highest_question': highest_question,
        'highest_score': highest_score,
        'total_questions': len(current_cols)
    }

# --- Course processing ---
def process_course(df, course, feedback_cols, ui_params):
    df_course = df[df['COURSE'] == course].copy()
    title_prefix = f"Course: {course}"
    
    if df_course.empty:
        st.warning(f"No data for {course}")
        return None

    current_cols = [col for col in feedback_cols if col in df_course.columns]
    if not current_cols:
        st.warning(f"No feedback columns for {course}")
        return None

    # Calculate and display KPIs
    kpis = calculate_kpis(df_course, feedback_cols)
    if kpis:
        st.markdown("### <i class='fas fa-chart-line'></i> Key Performance Indicators", unsafe_allow_html=True)
        
        # First row of KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-icon" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                        <i class="fas fa-users"></i>
                    </div>
                    <div class="kpi-content">
                        <div class="kpi-value">{kpis['total_responses']}</div>
                        <div class="kpi-label">Total Responses</div>
                        <div class="kpi-description">Number of students who provided feedback</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-icon" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                        <i class="fas fa-star"></i>
                    </div>
                    <div class="kpi-content">
                        <div class="kpi-value">{kpis['avg_score']:.2f}/5.0</div>
                        <div class="kpi-label">Average Score</div>
                        <div class="kpi-description">Mean rating across all questions</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-icon" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                        <i class="fas fa-smile"></i>
                    </div>
                    <div class="kpi-content">
                        <div class="kpi-value">{kpis['satisfaction_rate']:.1f}%</div>
                        <div class="kpi-label">Satisfaction Rate</div>
                        <div class="kpi-description">Percentage of 4-5 star ratings</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-icon" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                        <i class="fas fa-check-circle"></i>
                    </div>
                    <div class="kpi-content">
                        <div class="kpi-value">{kpis['response_rate']:.1f}%</div>
                        <div class="kpi-label">Response Rate</div>
                        <div class="kpi-description">Completion rate of all questions</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # Second row of KPIs
        st.markdown("<br>", unsafe_allow_html=True)
        col5, col6, col7 = st.columns([1, 1, 1])
        
        with col5:
            st.markdown(f"""
                <div class="kpi-card-wide">
                    <div class="kpi-icon-small" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
                        <i class="fas fa-arrow-up"></i>
                    </div>
                    <div class="kpi-content">
                        <div class="kpi-value-small">{kpis['highest_score']:.2f}</div>
                        <div class="kpi-label-small">Highest Score</div>
                        <div class="kpi-detail">{textwrap.shorten(str(kpis['highest_question']), width=40, placeholder="...")}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col6:
            st.markdown(f"""
                <div class="kpi-card-wide">
                    <div class="kpi-icon-small" style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);">
                        <i class="fas fa-arrow-down"></i>
                    </div>
                    <div class="kpi-content">
                        <div class="kpi-value-small">{kpis['lowest_score']:.2f}</div>
                        <div class="kpi-label-small">Lowest Score</div>
                        <div class="kpi-detail">{textwrap.shorten(str(kpis['lowest_question']), width=40, placeholder="...")}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col7:
            st.markdown(f"""
                <div class="kpi-card-wide">
                    <div class="kpi-icon-small" style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);">
                        <i class="fas fa-list-alt"></i>
                    </div>
                    <div class="kpi-content">
                        <div class="kpi-value-small">{kpis['total_questions']}</div>
                        <div class="kpi-label-small">Total Questions</div>
                        <div class="kpi-detail">Evaluated in survey</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

    df_numeric = df_course[current_cols].apply(pd.to_numeric, errors='coerce')
    df_long = df_numeric.melt(var_name='Question', value_name='Response').dropna()

    if not df_long.empty:
        st.markdown("### <i class='fas fa-chart-bar'></i> Detailed Analytics", unsafe_allow_html=True)
        
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

        flat = df_numeric.values.flatten()
        pct = calculate_cumulative_percentage(pd.Series(flat))
        st.info(f"📊 Cumulative Mean Percentage: **{pct:.2f}%**")
        plot_cumulative_pie(
            course, pct,
            fig_w=ui_params['pie_fig_w'], fig_h=ui_params['pie_fig_h'],
            donut_width=ui_params['pie_donut_width'],
            title_font=ui_params['pie_title_font'], pct_font=ui_params['pie_pct_font'],
            show_percentage=ui_params['pie_show_pct'],
            custom_title=ui_params['pie_title'],
            color_main=ui_params['pie_color_main'], color_bg=ui_params['pie_color_bg']
        )
        return True
    else:
        st.warning(f"No valid numeric responses for {course}")
        return None

# --- Streamlit UI ---
st.set_page_config(page_title="Teacher Feedback Analyzer", page_icon="📊", layout="wide")



# Custom CSS with Font Awesome and enhanced styling
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .course-nav {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .course-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid #e0e0e0;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.15);
    }
    .kpi-card-wide {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        transition: all 0.3s ease;
        border: 1px solid #e0e0e0;
        display: flex;
        align-items: center;
        gap: 1rem;
        min-height: 100px;
    }
    .kpi-card-wide:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.12);
    }
    .kpi-icon {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1rem auto;
        color: white;
        font-size: 24px;
    }
    .kpi-icon-small {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 20px;
        flex-shrink: 0;
    }
    .kpi-content {
        flex: 1;
    }
    .kpi-value {
        font-size: 32px;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.3rem;
    }
    .kpi-value-small {
        font-size: 24px;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.2rem;
    }
    .kpi-label {
        font-size: 14px;
        color: #7f8c8d;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    .kpi-label-small {
        font-size: 12px;
        color: #7f8c8d;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-description {
        font-size: 12px;
        color: #95a5a6;
        margin-top: 0.3rem;
        line-height: 1.4;
    }
    .kpi-detail {
        font-size: 11px;
        color: #7f8c8d;
        margin-top: 0.3rem;
        font-style: italic;
    }
    .progress-container {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1><i class="fas fa-chart-line"></i> Interactive Teacher Feedback Visualization Tool</h1><p>Upload CSV and customize charts with enhanced navigation and analytics</p>Developed By Subhradeep Sarkar, P241321, under the guidance of Dr. P Thiyagarajan, Department of Computer Science, CUTN<p></p></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        st.stop()

    if 'COURSE' not in df.columns:
        st.error("❌ CSV must contain a 'COURSE' column.")
        st.stop()

    df['COURSE'] = df['COURSE'].astype(str).str.strip()
    
    feedback_cols = df.columns[METADATA_END_INDEX + 1:].tolist()
    
    if not feedback_cols:
        st.error("❌ No feedback columns found.")
        st.stop()

    create_output_dir()

    # Use COURSE for navigation
    items_list = df['COURSE'].unique().tolist()
    navigation_field = 'COURSE'

    # Initialize session state for item index
    if 'course_index' not in st.session_state:
        st.session_state.course_index = 0
    
    # Ensure course_index is within bounds
    if st.session_state.course_index >= len(items_list):
        st.session_state.course_index = 0

    # --- Sidebar UI controls ---
    with st.sidebar:
        st.markdown("### <i class='fas fa-palette'></i> Chart Customization", unsafe_allow_html=True)
        
        with st.expander("📊 Distribution Chart", expanded=False):
            st.markdown("<i class='fas fa-chart-bar'></i> **Chart Settings**", unsafe_allow_html=True)
            dist_fig_w = st.slider("Width", 5, 20, 14, key="dist_w")
            dist_fig_h = st.slider("Height", 4, 15, 8, key="dist_h")
            dist_title_font = st.slider("Title font", 10, 30, 18, key="dist_tf")
            dist_label_font = st.slider("Axis label font", 8, 20, 12, key="dist_lf")
            dist_tick_font = st.slider("Tick label font", 6, 16, 10, key="dist_tick")
            dist_palette = st.selectbox("Color palette", ["viridis","magma","plasma","coolwarm"], key="dist_pal")
            dist_x_rotation = st.slider("X-axis rotation", 0, 90, 90, key="dist_rot")
            dist_y_max = st.slider("Y-axis max", 50, 150, 100, key="dist_ymax")
            dist_show_legend = st.checkbox("Show legend", True, key="dist_leg")
            dist_title = st.text_input("Custom title", "Response Distribution (% per Question)", key="dist_title")
            dist_xlabel = st.text_input("X-axis label", "Feedback Question", key="dist_xl")
            dist_ylabel = st.text_input("Y-axis label", "Percentage of Responses (%)", key="dist_yl")

        with st.expander("📈 Average Scores Chart", expanded=False):
            st.markdown("<i class='fas fa-chart-line'></i> **Chart Settings**", unsafe_allow_html=True)
            avg_fig_w = st.slider("Width", 5, 20, 12, key="avg_w")
            avg_fig_h = st.slider("Height", 4, 15, 10, key="avg_h")
            avg_title_font = st.slider("Title font", 10, 30, 16, key="avg_tf")
            avg_label_font = st.slider("Axis label font", 8, 20, 12, key="avg_lf")
            avg_tick_font = st.slider("Tick font", 6, 16, 10, key="avg_tick")
            avg_palette = st.selectbox("Color palette", ["viridis","magma","plasma","coolwarm"], index=0, key="avg_pal")
            avg_show_legend = st.checkbox("Show legend", True, key="avg_leg")
            avg_title = st.text_input("Custom title", "Average Scores", key="avg_title")
            avg_xlabel = st.text_input("X-axis label", "Average Score (1–5)", key="avg_xl")
            avg_ylabel = st.text_input("Y-axis label", "Feedback Question", key="avg_yl")

        with st.expander("🥧 Cumulative Pie Chart", expanded=False):
            st.markdown("<i class='fas fa-chart-pie'></i> **Chart Settings**", unsafe_allow_html=True)
            pie_fig_w = st.slider("Width", 4, 10, 6, key="pie_w")
            pie_fig_h = st.slider("Height", 4, 10, 6, key="pie_h")
            pie_donut_width = st.slider("Donut width", 0.1, 0.9, 0.4, key="pie_dw")
            pie_title_font = st.slider("Title font", 10, 30, 16, key="pie_tf")
            pie_pct_font = st.slider("Percentage font", 8, 24, 18, key="pie_pf")
            pie_show_pct = st.checkbox("Show percentage", True, key="pie_sp")
            pie_title = st.text_input("Custom title", "Cumulative Mean Score", key="pie_title")
            pie_color_main = st.color_picker("Main color", "#43a047", key="pie_cm")
            pie_color_bg = st.color_picker("Background color", "#e0e0e0", key="pie_cb")

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

    # Enhanced Navigation with Progress
    st.markdown('<div class="course-nav">', unsafe_allow_html=True)
    st.markdown("### <i class='fas fa-compass'></i> Navigation", unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns([1, 1, 3, 1, 1])
    
    with col1:
        if st.button("⏮️ First", use_container_width=True):
            st.session_state.course_index = 0
            st.rerun()
    
    with col2:
        if st.button("◀️ Previous", use_container_width=True):
            if st.session_state.course_index > 0:
                st.session_state.course_index -= 1
                st.rerun()
    
    with col3:
        selected_item = st.selectbox(
            f"Select {navigation_field}",
            items_list,
            index=st.session_state.course_index,
            key="item_selector"
    )

    # update index only if selection changed
    if selected_item != items_list[st.session_state.course_index]:
        st.session_state.course_index = items_list.index(selected_item)
        st.rerun()

    with col4:
        if st.button("Next ▶️", use_container_width=True):
            if st.session_state.course_index < len(items_list) - 1:
                st.session_state.course_index += 1
                st.rerun()

    with col5:
        if st.button("Last ⏭️", use_container_width=True):
            st.session_state.course_index = len(items_list) - 1
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # --- Show Selected Course Info ---
    st.markdown(f"""
        <div class="course-info">
            <h2><i class='fas fa-book-open'></i> {selected_item}</h2>
            <p>{navigation_field} {st.session_state.course_index + 1} of {len(items_list)}</p>
        </div>
    """, unsafe_allow_html=True)

    # --- Progress animation box ---
    progress_container = st.empty()
    progress_bar = progress_container.progress(0)

    progress_bar.progress(35)

    # --- Process course and generate charts ---
    before_figs = set(plt.get_fignums())
    ok = process_course(df, selected_item, feedback_cols, ui_params)
    after_figs = set(plt.get_fignums())
    new_figs = sorted(list(after_figs - before_figs))

    progress_bar.progress(80)

    # Fallback if process created no figs but updated existing
    if not new_figs:
        all_figs = sorted(plt.get_fignums())
        new_figs = all_figs[-3:] if len(all_figs) >= 1 else []

    new_figs = new_figs[-3:]

    progress_bar.progress(100)
    st.success("✔ Analysis Completed")

    # --- Export Download Section ---
    if new_figs:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:

            # Name the fig files
            names = []
            if len(new_figs) == 3:
                names = ["distribution", "average_scores", "cumulative_pie"]
            else:
                names = [f"chart_{i+1}" for i in range(len(new_figs))]

            for fig_num, name in zip(new_figs, names):
                fig = plt.figure(fig_num)
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format="png", bbox_inches="tight")
                img_buf.seek(0)
                zf.writestr(f"{sanitize_filename(selected_item)}_{name}.png", img_buf.read())

        zip_buffer.seek(0)
        st.download_button(
            label=f"📥 Download charts for {selected_item}",
            data=zip_buffer.getvalue(),
            file_name=f"{sanitize_filename(selected_item)}_charts.zip",
            mime="application/zip",
            use_container_width=True
        )
    else:
        st.info("ℹ️ No charts yet — select an item to analyze.")

st.markdown(
    "<br><hr><p style='text-align:center; color:gray;'>Developed by Subhradeep Sarkar, P241321, Under the Guidance of Dr. P Thiyagarajan, Department of Computer Science, CUTN © 2025</p>",
    unsafe_allow_html=True
)

