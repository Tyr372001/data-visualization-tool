import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import textwrap
import os
import re
import io
import zipfile
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

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
    return (score / 5) * 100  # assuming 5 is max score

def apply_custom_style(ax, params, chart_type):
    """Apply custom styling to matplotlib axes"""
    if params.get(f'{chart_type}_grid_show', False):
        ax.grid(True, alpha=params.get(f'{chart_type}_grid_alpha', 0.3), 
                linestyle=params.get(f'{chart_type}_grid_style', '-'))
    
    # Set background color
    if params.get(f'{chart_type}_bg_color'):
        ax.set_facecolor(params[f'{chart_type}_bg_color'])
    
    # Customize spines
    for spine in ax.spines.values():
        spine.set_linewidth(params.get(f'{chart_type}_spine_width', 1))
        spine.set_color(params.get(f'{chart_type}_spine_color', 'black'))

def add_value_annotations(ax, chart_type, show_values, value_format, value_position):
    """Add value annotations to charts"""
    if show_values:
        for container in ax.containers:
            labels = []
            for bar in container:
                height = bar.get_height() if chart_type == 'vertical' else bar.get_width()
                if value_format == 'integer':
                    labels.append(f'{int(height)}')
                elif value_format == 'one_decimal':
                    labels.append(f'{height:.1f}')
                elif value_format == 'two_decimal':
                    labels.append(f'{height:.2f}')
                else:
                    labels.append(f'{height}')
            
            if chart_type == 'vertical':
                ax.bar_label(container, labels=labels, padding=3, 
                           rotation=0 if value_position == 'horizontal' else 90)
            else:
                ax.bar_label(container, labels=labels, padding=4)

# --- Enhanced Plotting Functions ---
def plot_distribution(df_long, course, params):
    count_df = df_long.groupby(['Question','Response']).size().reset_index(name='Count')
    total_per_question = count_df.groupby('Question')['Count'].transform('sum')
    count_df['Percentage'] = count_df['Count'] / total_per_question * 100
    questions = list(pd.Categorical(count_df['Question']).categories)
    count_df['Question'] = pd.Categorical(count_df['Question'], categories=questions, ordered=True)

    fig, ax = plt.subplots(figsize=(params['dist_fig_w'], params['dist_fig_h']))
    
    # Create the plot
    if params['dist_plot_type'] == 'grouped':
        sns.barplot(
            data=count_df, x='Question', y='Percentage', hue='Response',
            palette=params['dist_palette'], ax=ax, order=questions,
            alpha=params['dist_alpha'], edgecolor=params['dist_edge_color'],
            linewidth=params['dist_edge_width']
        )
    elif params['dist_plot_type'] == 'stacked':
        # Pivot for stacked bar
        pivot_df = count_df.pivot(index='Question', columns='Response', values='Percentage').fillna(0)
        pivot_df.plot(kind='bar', stacked=True, ax=ax, 
                     color=sns.color_palette(params['dist_palette'], len(pivot_df.columns)),
                     alpha=params['dist_alpha'], edgecolor=params['dist_edge_color'],
                     linewidth=params['dist_edge_width'])

    # Customize labels
    wrapped_labels = ["\n".join(textwrap.wrap(str(q), params['dist_wrap_length'])) for q in questions]
    ax.set_xticklabels(wrapped_labels, rotation=params['dist_x_rotation'], 
                      ha=params['dist_x_align'], fontsize=params['dist_tick_font'],
                      fontweight=params['dist_tick_weight'])
    
    ax.set_xlabel(params['dist_xlabel'], fontsize=params['dist_label_font'], 
                 fontweight=params['dist_label_weight'], color=params['dist_label_color'])
    ax.set_ylabel(params['dist_ylabel'], fontsize=params['dist_label_font'],
                 fontweight=params['dist_label_weight'], color=params['dist_label_color'])
    ax.set_title(params['dist_title'], fontsize=params['dist_title_font'], 
                fontweight=params['dist_title_weight'], color=params['dist_title_color'], 
                pad=params['dist_title_pad'])
    ax.set_ylim(0, params['dist_y_max'])

    # Apply custom styling
    apply_custom_style(ax, params, 'dist')
    
    # Add value annotations
    if params['dist_show_values']:
        add_value_annotations(ax, 'vertical', True, params['dist_value_format'], 
                            params['dist_value_position'])

    # Legend customization
    if params['dist_show_legend']:
        legend = ax.legend(title=params['dist_legend_title'], 
                          bbox_to_anchor=params['dist_legend_position'], 
                          loc=params['dist_legend_loc'],
                          fontsize=params['dist_legend_fontsize'],
                          title_fontsize=params['dist_legend_title_fontsize'])
        legend.get_frame().set_facecolor(params['dist_legend_bg_color'])
        legend.get_frame().set_alpha(params['dist_legend_alpha'])
    else:
        if ax.get_legend():
            ax.get_legend().remove()

    plt.tight_layout()
    st.pyplot(fig)
    return fig

def plot_average_scores(mean_scores, course, params):
    fig, ax = plt.subplots(figsize=(params['avg_fig_w'], params['avg_fig_h']))
    
    # Sort based on user preference
    if params['avg_sort_order'] == 'ascending':
        mean_scores = mean_scores.sort_values()
    elif params['avg_sort_order'] == 'descending':
        mean_scores = mean_scores.sort_values(ascending=False)
    # 'original' keeps the original order
    
    # Create horizontal or vertical bars
    if params['avg_orientation'] == 'horizontal':
        bars = ax.barh(range(len(mean_scores)), mean_scores.values, 
                      color=sns.color_palette(params['avg_palette'], len(mean_scores)),
                      alpha=params['avg_alpha'], edgecolor=params['avg_edge_color'],
                      linewidth=params['avg_edge_width'])
        ax.set_yticks(range(len(mean_scores)))
        ax.set_yticklabels(["\n".join(textwrap.wrap(str(q), params['avg_wrap_length'])) 
                           for q in mean_scores.index],
                          fontsize=params['avg_tick_font'], fontweight=params['avg_tick_weight'])
        ax.set_xlim(params['avg_x_min'], params['avg_x_max'])
    else:
        bars = ax.bar(range(len(mean_scores)), mean_scores.values,
                     color=sns.color_palette(params['avg_palette'], len(mean_scores)),
                     alpha=params['avg_alpha'], edgecolor=params['avg_edge_color'],
                     linewidth=params['avg_edge_width'])
        ax.set_xticks(range(len(mean_scores)))
        ax.set_xticklabels(["\n".join(textwrap.wrap(str(q), params['avg_wrap_length'])) 
                           for q in mean_scores.index],
                          rotation=params['avg_x_rotation'], ha=params['avg_x_align'],
                          fontsize=params['avg_tick_font'], fontweight=params['avg_tick_weight'])
        ax.set_ylim(params['avg_y_min'], params['avg_y_max'])

    ax.set_xlabel(params['avg_xlabel'], fontsize=params['avg_label_font'],
                 fontweight=params['avg_label_weight'], color=params['avg_label_color'])
    ax.set_ylabel(params['avg_ylabel'], fontsize=params['avg_label_font'],
                 fontweight=params['avg_label_weight'], color=params['avg_label_color'])
    ax.set_title(params['avg_title'], fontsize=params['avg_title_font'],
                fontweight=params['avg_title_weight'], color=params['avg_title_color'],
                pad=params['avg_title_pad'])

    # Apply custom styling
    apply_custom_style(ax, params, 'avg')
    
    # Add value annotations
    if params['avg_show_values']:
        for i, (bar, value) in enumerate(zip(bars, mean_scores.values)):
            if params['avg_orientation'] == 'horizontal':
                x_pos = value + 0.02 if params['avg_value_position'] == 'outside' else value / 2
                y_pos = bar.get_y() + bar.get_height() / 2
            else:
                x_pos = bar.get_x() + bar.get_width() / 2
                y_pos = value + 0.02 if params['avg_value_position'] == 'outside' else value / 2
            
            if params['avg_value_format'] == 'integer':
                label = f'{int(value)}'
            elif params['avg_value_format'] == 'one_decimal':
                label = f'{value:.1f}'
            else:
                label = f'{value:.2f}'
                
            ax.text(x_pos, y_pos, label, ha='center', va='center',
                   fontsize=params['avg_value_fontsize'], fontweight='bold')

    # Add reference lines
    if params['avg_show_ref_line']:
        if params['avg_orientation'] == 'horizontal':
            ax.axvline(x=params['avg_ref_line_value'], color=params['avg_ref_line_color'],
                      linestyle=params['avg_ref_line_style'], linewidth=params['avg_ref_line_width'],
                      alpha=0.7, label=f"Reference ({params['avg_ref_line_value']})")
        else:
            ax.axhline(y=params['avg_ref_line_value'], color=params['avg_ref_line_color'],
                      linestyle=params['avg_ref_line_style'], linewidth=params['avg_ref_line_width'],
                      alpha=0.7, label=f"Reference ({params['avg_ref_line_value']})")
        
        if params['avg_show_ref_legend']:
            ax.legend(fontsize=params['avg_legend_fontsize'])

    plt.tight_layout()
    st.pyplot(fig)
    return fig

def plot_cumulative_pie(course, percent, params):
    score = max(min(percent, 100), 0)
    remainder = 100 - score
    fig, ax = plt.subplots(figsize=(params['pie_fig_w'], params['pie_fig_h']))
    
    if params['pie_style'] == 'donut':
        wedges, texts = ax.pie(
            [score, remainder], labels=['', ''], colors=[params['pie_color_main'], params['pie_color_bg']],
            startangle=params['pie_start_angle'], 
            wedgeprops={'width': params['pie_donut_width'], 'edgecolor': params['pie_edge_color'],
                       'linewidth': params['pie_edge_width']},
            explode=params['pie_explode'] if params['pie_explode_enable'] else None
        )
    elif params['pie_style'] == 'full_pie':
        wedges, texts = ax.pie(
            [score, remainder], labels=['', ''], colors=[params['pie_color_main'], params['pie_color_bg']],
            startangle=params['pie_start_angle'],
            wedgeprops={'edgecolor': params['pie_edge_color'], 'linewidth': params['pie_edge_width']},
            explode=params['pie_explode'] if params['pie_explode_enable'] else None
        )
    else:  # semi_circle
        wedges, texts = ax.pie(
            [score, remainder, remainder], labels=['', '', ''], 
            colors=[params['pie_color_main'], params['pie_color_bg'], 'white'],
            startangle=params['pie_start_angle'], 
            wedgeprops={'width': params['pie_donut_width'], 'edgecolor': params['pie_edge_color'],
                       'linewidth': params['pie_edge_width']}
        )

    # Center text
    if params['pie_show_pct']:
        main_text_y = 0.1 if params['pie_show_subtitle'] else 0
        ax.text(0, main_text_y, f'{score:.1f}%', ha='center', va='center',
                fontsize=params['pie_pct_font'], fontweight=params['pie_pct_weight'], 
                color=params['pie_pct_color'])
        
        if params['pie_show_subtitle']:
            ax.text(0, -0.2, params['pie_subtitle'], ha='center', va='center', 
                   fontsize=params['pie_subtitle_font'], color=params['pie_subtitle_color'])

    # Title
    ax.set_title(params['pie_title'], fontsize=params['pie_title_font'],
                fontweight=params['pie_title_weight'], color=params['pie_title_color'], 
                pad=params['pie_title_pad'])

    # Background
    if params['pie_bg_color']:
        fig.patch.set_facecolor(params['pie_bg_color'])
    
    plt.tight_layout()
    st.pyplot(fig)
    return fig

# --- Course Processing ---
def process_course(df, course, feedback_cols, ui_params):
    st.subheader(f"📊 Course Analysis: {course}")
    df_course = df[df['COURSE'] == course].copy()
    
    if df_course.empty:
        st.warning(f"⚠️ No data available for {course}")
        return

    current_cols = [col for col in feedback_cols if col in df_course.columns]
    if not current_cols:
        st.warning(f"⚠️ No feedback columns found for {course}")
        return

    df_numeric = df_course[current_cols].apply(pd.to_numeric, errors='coerce')
    
    # Display summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📋 Total Responses", len(df_course))
    with col2:
        st.metric("❓ Questions", len(current_cols))
    with col3:
        overall_mean = df_numeric.mean().mean()
        st.metric("📈 Overall Mean", f"{overall_mean:.2f}")
    with col4:
        response_rate = (df_numeric.count().sum() / (len(df_course) * len(current_cols))) * 100
        st.metric("✅ Response Rate", f"{response_rate:.1f}%")

    df_long = df_numeric.melt(var_name='Question', value_name='Response').dropna()

    if not df_long.empty:
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["📊 Response Distribution", "📈 Average Scores", "🎯 Overall Performance"])
        
        with tab1:
            st.markdown("### Response Distribution Analysis")
            plot_distribution(df_long, course, ui_params)
            
            # Show distribution summary
            with st.expander("📋 Distribution Summary"):
                dist_summary = df_long.groupby('Question')['Response'].agg(['mean', 'std', 'count'])
                st.dataframe(dist_summary.round(2))
        
        with tab2:
            st.markdown("### Average Scores by Question")
            mean_scores = df_numeric.mean().sort_values()
            plot_average_scores(mean_scores, course, ui_params)
            
            # Show top/bottom questions
            with st.expander("🏆 Top & Bottom Questions"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🎯 Highest Rated Questions:**")
                    top_questions = mean_scores.tail(3)
                    for q, score in top_questions.items():
                        st.write(f"• {q}: {score:.2f}")
                
                with col2:
                    st.markdown("**⚠️ Lowest Rated Questions:**")
                    bottom_questions = mean_scores.head(3)
                    for q, score in bottom_questions.items():
                        st.write(f"• {q}: {score:.2f}")
        
        with tab3:
            st.markdown("### Overall Performance Score")
            flat = df_numeric.values.flatten()
            pct = calculate_cumulative_percentage(pd.Series(flat))
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("🎯 Overall Score", f"{pct:.1f}%", 
                         delta=f"{pct-80:.1f}%" if pct >= 80 else None)
                
                # Performance interpretation
                if pct >= 90:
                    st.success("🌟 Excellent Performance!")
                elif pct >= 80:
                    st.success("✅ Good Performance")
                elif pct >= 70:
                    st.warning("⚠️ Needs Improvement")
                else:
                    st.error("❌ Requires Attention")
            
            with col2:
                plot_cumulative_pie(course, pct, ui_params)
    else:
        st.warning(f"⚠️ No valid numeric responses found for {course}")

# --- UI Parameter Functions ---
def get_distribution_params():
    with st.expander("📊 Distribution Chart Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎨 Visual Style**")
            plot_type = st.selectbox("Plot Type", ["grouped", "stacked"], key="dist_plot_type")
            palette = st.selectbox("Color Palette", ["viridis", "magma", "plasma", "coolwarm", "Set2", "tab10"], key="dist_palette")
            alpha = st.slider("Transparency", 0.1, 1.0, 0.8, key="dist_alpha")
            edge_color = st.color_picker("Edge Color", "#000000", key="dist_edge_color")
            edge_width = st.slider("Edge Width", 0.0, 3.0, 0.8, key="dist_edge_width")
            
            st.markdown("**📐 Layout**")
            fig_w = st.slider("Chart Width", 5, 20, 14, key="dist_fig_w")
            fig_h = st.slider("Chart Height", 4, 15, 8, key="dist_fig_h")
            y_max = st.slider("Y-axis Maximum", 50, 150, 100, key="dist_y_max")
            
        with col2:
            st.markdown("**📝 Text & Labels**")
            title = st.text_input("Chart Title", "Response Distribution (% per Question)", key="dist_title")
            xlabel = st.text_input("X-axis Label", "Feedback Question", key="dist_xlabel")
            ylabel = st.text_input("Y-axis Label", "Percentage of Responses (%)", key="dist_ylabel")
            
            # Font settings
            title_font = st.slider("Title Font Size", 10, 30, 18, key="dist_title_font")
            label_font = st.slider("Axis Label Font Size", 8, 20, 12, key="dist_label_font")
            tick_font = st.slider("Tick Font Size", 6, 16, 10, key="dist_tick_font")
            
            # Advanced text settings
            title_weight = st.selectbox("Title Weight", ["normal", "bold"], index=1, key="dist_title_weight")
            label_weight = st.selectbox("Label Weight", ["normal", "bold"], key="dist_label_weight")
            tick_weight = st.selectbox("Tick Weight", ["normal", "bold"], key="dist_tick_weight")
            
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**🎯 X-axis Customization**")
            x_rotation = st.slider("X-axis Rotation", 0, 90, 45, key="dist_x_rotation")
            x_align = st.selectbox("X-axis Alignment", ["center", "left", "right"], key="dist_x_align")
            wrap_length = st.slider("Label Wrap Length", 10, 50, 25, key="dist_wrap_length")
            
            st.markdown("**🎨 Colors**")
            title_color = st.color_picker("Title Color", "#000000", key="dist_title_color")
            label_color = st.color_picker("Label Color", "#000000", key="dist_label_color")
            bg_color = st.color_picker("Background Color", "#FFFFFF", key="dist_bg_color")
            
        with col4:
            st.markdown("**📊 Values & Legend**")
            show_values = st.checkbox("Show Values on Bars", key="dist_show_values")
            if show_values:
                value_format = st.selectbox("Value Format", ["integer", "one_decimal", "two_decimal"], 
                                          index=1, key="dist_value_format")
                value_position = st.selectbox("Value Position", ["horizontal", "vertical"], key="dist_value_position")
            
            show_legend = st.checkbox("Show Legend", True, key="dist_show_legend")
            if show_legend:
                legend_title = st.text_input("Legend Title", "Response (1-5)", key="dist_legend_title")
                legend_position = st.selectbox("Legend Position", 
                                             [(1.02, 0.5), (0.5, -0.1), (0.5, 1.1)], 
                                             format_func=lambda x: "Right" if x[0] > 1 else "Bottom" if x[1] < 0 else "Top",
                                             key="dist_legend_position")
                legend_loc = st.selectbox("Legend Location", ["center left", "upper center", "lower center"], 
                                        key="dist_legend_loc")
            
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            col1, col2 = st.columns(2)
            with col1:
                title_pad = st.slider("Title Padding", 5, 30, 16, key="avg_title_pad")
                grid_show = st.checkbox("Show Grid", key="avg_grid_show")
                if grid_show:
                    grid_alpha = st.slider("Grid Transparency", 0.1, 1.0, 0.3, key="avg_grid_alpha")
                    grid_style = st.selectbox("Grid Style", ["-", "--", "-.", ":"], key="avg_grid_style")
            
            with col2:
                spine_width = st.slider("Border Width", 0.5, 3.0, 1.0, key="avg_spine_width")
                spine_color = st.color_picker("Border Color", "#000000", key="avg_spine_color")

def get_pie_params():
    with st.expander("🎯 Pie Chart Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎨 Visual Style**")
            pie_style = st.selectbox("Pie Style", ["donut", "full_pie", "semi_circle"], key="pie_style")
            if pie_style in ["donut", "semi_circle"]:
                donut_width = st.slider("Donut Width", 0.1, 0.9, 0.4, key="pie_donut_width")
            
            color_main = st.color_picker("Main Color", "#43a047", key="pie_color_main")
            color_bg = st.color_picker("Background Color", "#e0e0e0", key="pie_color_bg")
            edge_color = st.color_picker("Edge Color", "#FFFFFF", key="pie_edge_color")
            edge_width = st.slider("Edge Width", 0.0, 3.0, 2.0, key="pie_edge_width")
            
            st.markdown("**📐 Layout**")
            fig_w = st.slider("Chart Width", 4, 12, 8, key="pie_fig_w")
            fig_h = st.slider("Chart Height", 4, 12, 8, key="pie_fig_h")
            start_angle = st.slider("Start Angle", 0, 360, 90, key="pie_start_angle")
            
        with col2:
            st.markdown("**📝 Text & Labels**")
            title = st.text_input("Chart Title", "Overall Performance Score", key="pie_title")
            title_font = st.slider("Title Font Size", 10, 30, 18, key="pie_title_font")
            title_weight = st.selectbox("Title Weight", ["normal", "bold"], index=1, key="pie_title_weight")
            title_color = st.color_picker("Title Color", "#000000", key="pie_title_color")
            title_pad = st.slider("Title Padding", 5, 30, 20, key="pie_title_pad")
            
            st.markdown("**🔢 Center Text**")
            show_pct = st.checkbox("Show Percentage", True, key="pie_show_pct")
            if show_pct:
                pct_font = st.slider("Percentage Font Size", 8, 36, 24, key="pie_pct_font")
                pct_weight = st.selectbox("Percentage Weight", ["normal", "bold"], index=1, key="pie_pct_weight")
                pct_color = st.color_picker("Percentage Color", "#000000", key="pie_pct_color")
                
                show_subtitle = st.checkbox("Show Subtitle", True, key="pie_show_subtitle")
                if show_subtitle:
                    subtitle = st.text_input("Subtitle Text", "Mean Score", key="pie_subtitle")
                    subtitle_font = st.slider("Subtitle Font Size", 8, 20, 14, key="pie_subtitle_font")
                    subtitle_color = st.color_picker("Subtitle Color", "#666666", key="pie_subtitle_color")
        
        # Advanced settings
        with st.expander("🔧 Advanced Pie Settings"):
            col1, col2 = st.columns(2)
            with col1:
                explode_enable = st.checkbox("Enable Explode Effect", key="pie_explode_enable")
                if explode_enable:
                    explode_main = st.slider("Main Slice Explode", 0.0, 0.3, 0.1, key="pie_explode_main")
                    explode_bg = st.slider("Background Slice Explode", 0.0, 0.3, 0.0, key="pie_explode_bg")
                    # Create explode tuple
                    st.session_state['pie_explode'] = (explode_main, explode_bg)
                
                bg_color = st.color_picker("Chart Background", "#FFFFFF", key="pie_bg_color")
            
            with col2:
                # Animation and effects could be added here
                shadow_enable = st.checkbox("Enable Shadow", key="pie_shadow_enable")
                if shadow_enable:
                    shadow_offset = st.slider("Shadow Offset", 0.01, 0.1, 0.02, key="pie_shadow_offset")

def get_global_params():
    with st.expander("🌐 Global Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎨 Theme Settings**")
            theme = st.selectbox("Color Theme", ["default", "dark", "colorful", "minimal", "academic"], key="global_theme")
            
            st.markdown("**💾 Export Settings**")
            export_dpi = st.slider("Export DPI", 72, 300, 150, key="export_dpi")
            export_format = st.selectbox("Export Format", ["PNG", "PDF", "SVG", "JPEG"], key="export_format")
            export_transparent = st.checkbox("Transparent Background", key="export_transparent")
            
        with col2:
            st.markdown("**📊 Data Settings**")
            filter_incomplete = st.checkbox("Filter Incomplete Responses", key="filter_incomplete")
            min_responses = st.slider("Minimum Responses per Question", 1, 20, 3, key="min_responses")
            
            st.markdown("**🔍 Analysis Options**")
            show_outliers = st.checkbox("Highlight Outliers", key="show_outliers")
            confidence_interval = st.checkbox("Show Confidence Intervals", key="show_confidence_interval")

# --- Main Streamlit App ---
def main():
    st.set_page_config(
        page_title="Enhanced Teacher Feedback Analyzer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 Enhanced Teacher Feedback Visualization Tool")
    st.markdown("""
    **Welcome to the comprehensive feedback analysis platform!** 
    Upload your CSV file and explore interactive visualizations with extensive customization options.
    """)
    
    # File upload section
    with st.container():
        st.markdown("### 📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Choose your CSV file", 
            type=["csv"],
            help="Upload a CSV file containing teacher feedback data with a 'COURSE' column."
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Successfully loaded {len(df)} records from {uploaded_file.name}")
                
                # Show data preview
                with st.expander("👁️ Data Preview"):
                    st.dataframe(df.head(), use_container_width=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Records", len(df))
                    with col2:
                        st.metric("Total Columns", len(df.columns))
                    with col3:
                        if 'COURSE' in df.columns:
                            st.metric("Unique Courses", df['COURSE'].nunique())
                
            except Exception as e:
                st.error(f"❌ Error loading CSV: {e}")
                st.stop()
            
            # Validate required columns
            if 'COURSE' not in df.columns:
                st.error("❌ CSV must contain a 'COURSE' column.")
                st.stop()
            
            df['COURSE'] = df['COURSE'].astype(str).str.strip()
            feedback_cols = df.columns[METADATA_END_INDEX + 1:].tolist()
            
            if not feedback_cols:
                st.error("❌ No feedback columns found after index 8.")
                st.stop()
            
            create_output_dir()
            
            # Sidebar customization panels
            st.sidebar.markdown("# 🎛️ Customization Panel")
            st.sidebar.markdown("Configure your visualizations using the sections below:")
            
            # Get all parameters from UI
            get_distribution_params()
            get_average_params()  
            get_pie_params()
            get_global_params()
            
            # Collect all parameters into a dictionary
            ui_params = {}
            for key in st.session_state:
                if any(key.startswith(prefix) for prefix in ['dist_', 'avg_', 'pie_', 'global_', 'export_']):
                    ui_params[key] = st.session_state[key]
            
            # Set defaults for any missing parameters
            defaults = {
                'dist_plot_type': 'grouped', 'dist_palette': 'viridis', 'dist_alpha': 0.8,
                'dist_edge_color': '#000000', 'dist_edge_width': 0.8, 'dist_fig_w': 14, 'dist_fig_h': 8,
                'dist_y_max': 100, 'dist_title': 'Response Distribution (% per Question)',
                'dist_xlabel': 'Feedback Question', 'dist_ylabel': 'Percentage of Responses (%)',
                'dist_title_font': 18, 'dist_label_font': 12, 'dist_tick_font': 10,
                'dist_title_weight': 'bold', 'dist_label_weight': 'normal', 'dist_tick_weight': 'normal',
                'dist_x_rotation': 45, 'dist_x_align': 'center', 'dist_wrap_length': 25,
                'dist_title_color': '#000000', 'dist_label_color': '#000000', 'dist_bg_color': '#FFFFFF',
                'dist_show_values': False, 'dist_value_format': 'one_decimal', 'dist_value_position': 'horizontal',
                'dist_show_legend': True, 'dist_legend_title': 'Response (1-5)', 'dist_legend_position': (1.02, 0.5),
                'dist_legend_loc': 'center left', 'dist_title_pad': 16, 'dist_grid_show': False,
                'dist_grid_alpha': 0.3, 'dist_grid_style': '-', 'dist_spine_width': 1.0, 'dist_spine_color': '#000000',
                'dist_legend_fontsize': 10, 'dist_legend_title_fontsize': 12, 'dist_legend_bg_color': '#FFFFFF',
                'dist_legend_alpha': 0.8,
                
                'avg_orientation': 'horizontal', 'avg_palette': 'viridis', 'avg_sort_order': 'ascending',
                'avg_alpha': 0.8, 'avg_edge_color': '#000000', 'avg_edge_width': 0.8,
                'avg_fig_w': 12, 'avg_fig_h': 10, 'avg_title': 'Average Scores by Question',
                'avg_xlabel': 'Average Score (1–5)', 'avg_ylabel': 'Feedback Question',
                'avg_title_font': 16, 'avg_label_font': 12, 'avg_tick_font': 10,
                'avg_title_weight': 'bold', 'avg_label_weight': 'normal', 'avg_tick_weight': 'normal',
                'avg_x_min': 1.0, 'avg_x_max': 5.0, 'avg_y_min': 1.0, 'avg_y_max': 5.0,
                'avg_x_rotation': 45, 'avg_x_align': 'center', 'avg_wrap_length': 30,
                'avg_title_color': '#000000', 'avg_label_color': '#000000', 'avg_bg_color': '#FFFFFF',
                'avg_show_values': True, 'avg_value_format': 'two_decimal', 'avg_value_position': 'outside',
                'avg_value_fontsize': 10, 'avg_show_ref_line': False, 'avg_ref_line_value': 3.0,
                'avg_ref_line_color': '#FF0000', 'avg_ref_line_style': '--', 'avg_ref_line_width': 2.0,
                'avg_show_ref_legend': False, 'avg_legend_fontsize': 10, 'avg_title_pad': 16,
                'avg_grid_show': False, 'avg_grid_alpha': 0.3, 'avg_grid_style': '-',
                'avg_spine_width': 1.0, 'avg_spine_color': '#000000',
                
                'pie_style': 'donut', 'pie_donut_width': 0.4, 'pie_color_main': '#43a047',
                'pie_color_bg': '#e0e0e0', 'pie_edge_color': '#FFFFFF', 'pie_edge_width': 2.0,
                'pie_fig_w': 8, 'pie_fig_h': 8, 'pie_start_angle': 90,
                'pie_title': 'Overall Performance Score', 'pie_title_font': 18, 'pie_title_weight': 'bold',
                'pie_title_color': '#000000', 'pie_title_pad': 20, 'pie_show_pct': True,
                'pie_pct_font': 24, 'pie_pct_weight': 'bold', 'pie_pct_color': '#000000',
                'pie_show_subtitle': True, 'pie_subtitle': 'Mean Score', 'pie_subtitle_font': 14,
                'pie_subtitle_color': '#666666', 'pie_explode_enable': False, 'pie_explode': (0.0, 0.0),
                'pie_bg_color': '#FFFFFF', 'pie_shadow_enable': False, 'pie_shadow_offset': 0.02
            }
            
            for key, default_value in defaults.items():
                if key not in ui_params:
                    ui_params[key] = default_value
            
            # Course selection and analysis
            st.markdown("### 🎓 Course Analysis")
            
            courses = sorted(df['COURSE'].unique())
            
            # Analysis mode selection
            analysis_mode = st.radio(
                "Choose Analysis Mode:",
                ["Single Course", "Compare Courses", "All Courses"],
                horizontal=True
            )
            
            if analysis_mode == "Single Course":
                selected_course = st.selectbox("📚 Select Course to Analyze:", courses)
                
                if st.button("🚀 Generate Analysis", type="primary"):
                    with st.spinner("Generating comprehensive analysis..."):
                        process_course(df, selected_course, feedback_cols, ui_params)
                        
                        # Generate download package
                        _before_figs = set(plt.get_fignums())
                        # Charts are already generated in process_course
                        _after_figs = set(plt.get_fignums())
                        _new_figs = sorted(list(_after_figs - _before_figs))
                        
                        if not _new_figs:
                            _all_figs = sorted(plt.get_fignums())
                            _new_figs = _all_figs[-3:] if len(_all_figs) >= 3 else _all_figs
                        
                        if _new_figs:
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                                chart_names = ["distribution", "average_scores", "overall_performance"]
                                
                                for i, fig_num in enumerate(_new_figs[-3:]):
                                    chart_name = chart_names[i] if i < len(chart_names) else f"chart_{i+1}"
                                    fig = plt.figure(fig_num)
                                    img_buf = io.BytesIO()
                                    
                                    # Use export settings
                                    export_format = ui_params.get('export_format', 'PNG').lower()
                                    if export_format == 'jpeg':
                                        export_format = 'jpg'
                                    
                                    fig.savefig(
                                        img_buf, 
                                        format=export_format,
                                        dpi=ui_params.get('export_dpi', 150),
                                        bbox_inches="tight",
                                        transparent=ui_params.get('export_transparent', False)
                                    )
                                    img_buf.seek(0)
                                    
                                    filename = f"{sanitize_filename(selected_course)}_{chart_name}.{export_format}"
                                    zf.writestr(filename, img_buf.read())
                            
                            zip_buffer.seek(0)
                            st.download_button(
                                label=f"📥 Download Complete Analysis Package",
                                data=zip_buffer.getvalue(),
                                file_name=f"{sanitize_filename(selected_course)}_analysis_package.zip",
                                mime="application/zip",
                                help="Downloads all charts and analysis data for the selected course"
                            )
            
            elif analysis_mode == "Compare Courses":
                selected_courses = st.multiselect("📊 Select Courses to Compare (2-5):", courses, max_selections=5)
                
                if len(selected_courses) >= 2:
                    if st.button("🔍 Generate Comparison", type="primary"):
                        st.markdown("### 📊 Course Comparison Analysis")
                        
                        # Create comparison dataframe
                        comparison_data = []
                        for course in selected_courses:
                            df_course = df[df['COURSE'] == course]
                            current_cols = [col for col in feedback_cols if col in df_course.columns]
                            df_numeric = df_course[current_cols].apply(pd.to_numeric, errors='coerce')
                            
                            overall_mean = df_numeric.mean().mean()
                            response_count = len(df_course)
                            response_rate = (df_numeric.count().sum() / (len(df_course) * len(current_cols))) * 100
                            
                            comparison_data.append({
                                'Course': course,
                                'Overall Mean': overall_mean,
                                'Response Count': response_count,
                                'Response Rate (%)': response_rate,
                                'Highest Score': df_numeric.mean().max(),
                                'Lowest Score': df_numeric.mean().min()
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        
                        # Display comparison table
                        st.markdown("#### 📈 Comparison Summary")
                        st.dataframe(comparison_df.round(2), use_container_width=True)
                        
                        # Create comparison visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            bars = ax.bar(comparison_df['Course'], comparison_df['Overall Mean'], 
                                         color=sns.color_palette("viridis", len(comparison_df)))
                            ax.set_title('Overall Mean Scores Comparison', fontsize=16, fontweight='bold')
                            ax.set_ylabel('Mean Score')
                            ax.set_ylim(0, 5)
                            
                            for bar, value in zip(bars, comparison_df['Overall Mean']):
                                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                                       f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
                            
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        with col2:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.scatter(comparison_df['Response Count'], comparison_df['Overall Mean'], 
                                      s=200, alpha=0.7, c=sns.color_palette("viridis", len(comparison_df)))
                            
                            for i, course in enumerate(comparison_df['Course']):
                                ax.annotate(course, 
                                          (comparison_df['Response Count'].iloc[i], 
                                           comparison_df['Overall Mean'].iloc[i]),
                                          xytext=(5, 5), textcoords='offset points', fontsize=10)
                            
                            ax.set_xlabel('Number of Responses')
                            ax.set_ylabel('Overall Mean Score')
                            ax.set_title('Response Count vs Mean Score', fontsize=16, fontweight='bold')
                            plt.tight_layout()
                            st.pyplot(fig)
                
                else:
                    st.info("👆 Please select at least 2 courses to compare.")
            
            else:  # All Courses
                if st.button("📊 Generate All Courses Report", type="primary"):
                    st.markdown("### 📋 Complete Institutional Analysis")
                    
                    with st.spinner("Generating comprehensive institutional report..."):
                        # Create summary statistics
                        all_course_data = []
                        
                        for course in courses:
                            df_course = df[df['COURSE'] == course]
                            current_cols = [col for col in feedback_cols if col in df_course.columns]
                            
                            if current_cols:
                                df_numeric = df_course[current_cols].apply(pd.to_numeric, errors='coerce')
                                
                                overall_mean = df_numeric.mean().mean()
                                response_count = len(df_course)
                                response_rate = (df_numeric.count().sum() / (len(df_course) * len(current_cols))) * 100
                                
                                all_course_data.append({
                                    'Course': course,
                                    'Overall Mean': overall_mean,
                                    'Response Count': response_count,
                                    'Response Rate (%)': response_rate,
                                    'Performance Level': 'Excellent' if overall_mean >= 4.5 else 
                                                       'Good' if overall_mean >= 4.0 else
                                                       'Satisfactory' if overall_mean >= 3.5 else
                                                       'Needs Improvement'
                                })
                        
                        summary_df = pd.DataFrame(all_course_data)
                        
                        # Display institutional overview
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("📚 Total Courses", len(summary_df))
                        with col2:
                            st.metric("🎯 Institution Average", f"{summary_df['Overall Mean'].mean():.2f}")
                        with col3:
                            excellent_count = len(summary_df[summary_df['Performance Level'] == 'Excellent'])
                            st.metric("🌟 Excellent Courses", f"{excellent_count} ({excellent_count/len(summary_df)*100:.1f}%)")
                        with col4:
                            total_responses = summary_df['Response Count'].sum()
                            st.metric("📝 Total Responses", f"{total_responses:,}")
                        
                        # Performance distribution
                        st.markdown("#### 📊 Performance Distribution")
                        performance_counts = summary_df['Performance Level'].value_counts()
                        
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                        
                        # Performance distribution pie chart
                        colors = ['#2E8B57', '#32CD32', '#FFD700', '#FF6347']
                        ax1.pie(performance_counts.values, labels=performance_counts.index, autopct='%1.1f%%',
                               colors=colors, startangle=90)
                        ax1.set_title('Performance Level Distribution', fontsize=14, fontweight='bold')
                        
                        # Overall mean distribution histogram
                        ax2.hist(summary_df['Overall Mean'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                        ax2.set_xlabel('Overall Mean Score')
                        ax2.set_ylabel('Number of Courses')
                        ax2.set_title('Distribution of Course Mean Scores', fontsize=14, fontweight='bold')
                        ax2.axvline(summary_df['Overall Mean'].mean(), color='red', linestyle='--', 
                                   label=f'Institution Average: {summary_df["Overall Mean"].mean():.2f}')
                        ax2.legend()
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Detailed course ranking
                        st.markdown("#### 🏆 Course Performance Ranking")
                        
                        # Sort by overall mean descending
                        ranked_df = summary_df.sort_values('Overall Mean', ascending=False).reset_index(drop=True)
                        ranked_df.index += 1  # Start ranking from 1
                        
                        st.dataframe(ranked_df, use_container_width=True)
                        
                        # Top and bottom performers
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**🥇 Top 5 Performing Courses:**")
                            top_5 = ranked_df.head(5)
                            for idx, row in top_5.iterrows():
                                st.write(f"{idx}. **{row['Course']}** - {row['Overall Mean']:.2f} ⭐")
                        
                        with col2:
                            st.markdown("**⚠️ Courses Needing Attention:**")
                            bottom_5 = ranked_df.tail(5)
                            for idx, row in bottom_5.iterrows():
                                st.write(f"{idx}. **{row['Course']}** - {row['Overall Mean']:.2f}")
                        
                        # Generate institutional report download
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                            # Save the summary data as CSV
                            csv_buffer = io.StringIO()
                            ranked_df.to_csv(csv_buffer, index_label='Rank')
                            zf.writestr("institutional_summary.csv", csv_buffer.getvalue().encode())
                            
                            # Save current figures
                            current_figs = plt.get_fignums()
                            if current_figs:
                                fig = plt.figure(current_figs[-1])  # Get the latest figure
                                img_buf = io.BytesIO()
                                fig.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
                                img_buf.seek(0)
                                zf.writestr("institutional_overview.png", img_buf.read())
                        
                        zip_buffer.seek(0)
                        st.download_button(
                            label="📥 Download Institutional Report Package",
                            data=zip_buffer.getvalue(),
                            file_name="institutional_feedback_report.zip",
                            mime="application/zip",
                            help="Downloads complete institutional analysis including data and visualizations"
                        )

        else:
            st.info("👆 Please upload a CSV file to begin analysis.")
            
            # Show example data format
            with st.expander("📋 Expected Data Format"):
                st.markdown("""
                Your CSV should have the following structure:
                
                - **COURSE**: Column containing course identifiers
                - **Columns 1-8**: Metadata columns (will be skipped)
                - **Columns 9+**: Feedback questions with numeric responses (1-5 scale)
                
                **Example:**
                ```
                COURSE,Instructor,Semester,...,Question1,Question2,Question3,...
                MATH101,Dr. Smith,Fall2023,...,4,5,3,...
                PHYS201,Prof. Jones,Fall2023,...,3,4,4,...
                ```
                """)

if __name__ == "__main__":
    main()d = st.slider("Title Padding", 5, 30, 16, key="dist_title_pad")
                grid_show = st.checkbox("Show Grid", key="dist_grid_show")
                if grid_show:
                    grid_alpha = st.slider("Grid Transparency", 0.1, 1.0, 0.3, key="dist_grid_alpha")
                    grid_style = st.selectbox("Grid Style", ["-", "--", "-.", ":"], key="dist_grid_style")
            
            with col2:
                spine_width = st.slider("Border Width", 0.5, 3.0, 1.0, key="dist_spine_width")
                spine_color = st.color_picker("Border Color", "#000000", key="dist_spine_color")
                if show_legend:
                    legend_fontsize = st.slider("Legend Font Size", 6, 16, 10, key="dist_legend_fontsize")
                    legend_title_fontsize = st.slider("Legend Title Font Size", 8, 18, 12, key="dist_legend_title_fontsize")
                    legend_bg_color = st.color_picker("Legend Background", "#FFFFFF", key="dist_legend_bg_color")
                    legend_alpha = st.slider("Legend Transparency", 0.0, 1.0, 0.8, key="dist_legend_alpha")

def get_average_params():
    with st.expander("📈 Average Scores Chart Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎨 Visual Style**")
            orientation = st.selectbox("Orientation", ["horizontal", "vertical"], key="avg_orientation")
            palette = st.selectbox("Color Palette", ["viridis", "magma", "plasma", "coolwarm", "Set2", "tab10"], key="avg_palette")
            sort_order = st.selectbox("Sort Order", ["ascending", "descending", "original"], key="avg_sort_order")
            alpha = st.slider("Transparency", 0.1, 1.0, 0.8, key="avg_alpha")
            edge_color = st.color_picker("Edge Color", "#000000", key="avg_edge_color")
            edge_width = st.slider("Edge Width", 0.0, 3.0, 0.8, key="avg_edge_width")
            
            st.markdown("**📐 Layout**")
            fig_w = st.slider("Chart Width", 5, 20, 12, key="avg_fig_w")
            fig_h = st.slider("Chart Height", 4, 15, 10, key="avg_fig_h")
            
        with col2:
            st.markdown("**📝 Text & Labels**")
            title = st.text_input("Chart Title", "Average Scores by Question", key="avg_title")
            xlabel = st.text_input("X-axis Label", "Average Score (1–5)", key="avg_xlabel")
            ylabel = st.text_input("Y-axis Label", "Feedback Question", key="avg_ylabel")
            
            # Font settings
            title_font = st.slider("Title Font Size", 10, 30, 16, key="avg_title_font")
            label_font = st.slider("Axis Label Font Size", 8, 20, 12, key="avg_label_font")
            tick_font = st.slider("Tick Font Size", 6, 16, 10, key="avg_tick_font")
            
            # Advanced text settings
            title_weight = st.selectbox("Title Weight", ["normal", "bold"], index=1, key="avg_title_weight")
            label_weight = st.selectbox("Label Weight", ["normal", "bold"], key="avg_label_weight")
            tick_weight = st.selectbox("Tick Weight", ["normal", "bold"], key="avg_tick_weight")
            
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**📏 Axis Ranges**")
            if orientation == "horizontal":
                x_min = st.slider("X-axis Minimum", 0.0, 3.0, 1.0, key="avg_x_min")
                x_max = st.slider("X-axis Maximum", 3.0, 5.0, 5.0, key="avg_x_max")
            else:
                y_min = st.slider("Y-axis Minimum", 0.0, 3.0, 1.0, key="avg_y_min")
                y_max = st.slider("Y-axis Maximum", 3.0, 5.0, 5.0, key="avg_y_max")
                x_rotation = st.slider("X-axis Rotation", 0, 90, 45, key="avg_x_rotation")
                x_align = st.selectbox("X-axis Alignment", ["center", "left", "right"], key="avg_x_align")
            
            wrap_length = st.slider("Label Wrap Length", 10, 50, 30, key="avg_wrap_length")
            
        with col4:
            st.markdown("**🎨 Colors & Style**")
            title_color = st.color_picker("Title Color", "#000000", key="avg_title_color")
            label_color = st.color_picker("Label Color", "#000000", key="avg_label_color")
            bg_color = st.color_picker("Background Color", "#FFFFFF", key="avg_bg_color")
            
            st.markdown("**📊 Values Display**")
            show_values = st.checkbox("Show Values", True, key="avg_show_values")
            if show_values:
                value_format = st.selectbox("Value Format", ["integer", "one_decimal", "two_decimal"], 
                                          index=2, key="avg_value_format")
                value_position = st.selectbox("Value Position", ["inside", "outside"], key="avg_value_position")
                value_fontsize = st.slider("Value Font Size", 6, 16, 10, key="avg_value_fontsize")
        
        # Reference line settings
        with st.expander("📏 Reference Line Settings"):
            col1, col2 = st.columns(2)
            with col1:
                show_ref_line = st.checkbox("Show Reference Line", key="avg_show_ref_line")
                if show_ref_line:
                    ref_line_value = st.slider("Reference Value", 1.0, 5.0, 3.0, key="avg_ref_line_value")
                    ref_line_color = st.color_picker("Reference Line Color", "#FF0000", key="avg_ref_line_color")
            
            with col2:
                if show_ref_line:
                    ref_line_style = st.selectbox("Reference Line Style", ["--", "-", "-.", ":"], key="avg_ref_line_style")
                    ref_line_width = st.slider("Reference Line Width", 0.5, 3.0, 2.0, key="avg_ref_line_width")
                    show_ref_legend = st.checkbox("Show Reference Legend", key="avg_show_ref_legend")
                    if show_ref_legend:
                        legend_fontsize = st.slider("Legend Font Size", 6, 16, 10, key="avg_legend_fontsize")
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            col1, col2 = st.columns(2)
            with col1:
                title_pa
