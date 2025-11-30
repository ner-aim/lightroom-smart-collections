"""
Smart Collections Intelligence System - Streamlit Dashboard

Professional Adobe-inspired interface for Lightroom catalog analysis.
Clean design with Lightroom Classic color palette and typography.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import sys
from pathlib import Path

# Import our modules
from generate_data import LightroomCatalogGenerator
from analysis import CatalogAnalyzer
from recommendations import MLRecommendationEngine

# Page configuration
st.set_page_config(
    page_title="Smart Collections Intelligence | Lightroom",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Adobe Lightroom-inspired CSS
st.markdown("""
    <style>
    /* Adobe Color Palette */
    :root {
        --adobe-blue: #1473E6;
        --adobe-dark: #2C2C2C;
        --adobe-gray: #505050;
        --adobe-light-gray: #E8E8E8;
        --adobe-success: #268E6C;
        --adobe-warning: #E68619;
        --adobe-error: #D7373F;
    }
    
    /* Main background - Lightroom dark theme */
    .main {
        background-color: #242424;
        color: #E8E8E8;
    }
    
    /* Headers */
    h1 {
        color: #FFFFFF;
        font-family: 'Adobe Clean', 'Segoe UI', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #404040;
    }
    
    h2 {
        color: #E8E8E8;
        font-family: 'Adobe Clean', 'Segoe UI', sans-serif;
        font-weight: 600;
        font-size: 1.75rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #D0D0D0;
        font-family: 'Adobe Clean', 'Segoe UI', sans-serif;
        font-weight: 600;
        font-size: 1.25rem;
        margin-top: 1.5rem;
    }
    
    /* Metrics - Lightroom style with rounded edges */
    .stMetric {
        background: linear-gradient(135deg, #2C2C2C 0%, #383838 100%);
        padding: 1.25rem;
        border-radius: 8px;
        border-left: 3px solid #1473E6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    .stMetric label {
        color: #B8B8B8 !important;
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stMetric div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .stMetric div[data-testid="stMetricDelta"] {
        color: #1473E6 !important;
        font-size: 0.875rem;
    }
    
    /* Buttons - Adobe style with smooth transitions */
    .stButton button {
        background: linear-gradient(135deg, #1473E6 0%, #0D66D0 100%);
        color: #FFFFFF;
        border: none;
        border-radius: 20px;
        padding: 0.625rem 1.75rem;
        font-weight: 600;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 2px 8px rgba(20, 115, 230, 0.3);
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #0D66D0 0%, #095ABA 100%);
        box-shadow: 0 4px 12px rgba(20, 115, 230, 0.5);
        transform: translateY(-2px);
    }
    
    .stButton button:active {
        transform: translateY(0);
        box-shadow: 0 2px 6px rgba(20, 115, 230, 0.4);
    }
    
    /* Alerts - Lightroom style with rounded corners */
    .stAlert {
        background-color: #2C2C2C;
        border-radius: 8px;
        border-left: 4px solid #1473E6;
        padding: 1rem;
        color: #E8E8E8;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    }
    
    /* Info boxes */
    div[data-baseweb="notification"] {
        background: linear-gradient(135deg, #2C2C2C 0%, #323232 100%);
        border-left: 4px solid #1473E6;
        border-radius: 8px;
        color: #E8E8E8;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A1A1A 0%, #242424 100%);
        border-right: 1px solid #404040;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #E8E8E8;
    }
    
    /* Radio buttons in sidebar */
    .stRadio label {
        color: #D0D0D0 !important;
        font-weight: 500;
    }
    
    .stRadio div[role="radiogroup"] label:hover {
        color: #1473E6 !important;
    }
    
    /* Expander - Smooth rounded design */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #2C2C2C 0%, #323232 100%);
        color: #E8E8E8;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #383838 0%, #3D3D3D 100%);
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    }
    
    .streamlit-expanderContent {
        background: #242424;
        border-radius: 0 0 8px 8px;
        padding: 1rem;
    }
    
    /* Code blocks - Rounded with subtle glow */
    .stCodeBlock {
        background: linear-gradient(135deg, #1A1A1A 0%, #1E1E1E 100%);
        border: 1px solid #404040;
        border-radius: 8px;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.3);
    }
    
    code {
        color: #1473E6;
        background-color: #1A1A1A;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        font-family: 'Consolas', 'Monaco', monospace;
    }
    
    /* Progress bars - Smooth and rounded */
    .stProgress > div > div {
        background: linear-gradient(90deg, #1473E6 0%, #0D66D0 100%);
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(20, 115, 230, 0.3);
    }
    
    .stProgress > div {
        background-color: #1A1A1A;
        border-radius: 10px;
    }
    
    /* Dataframes */
    .stDataFrame {
        background-color: #2C2C2C;
        color: #E8E8E8;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #2C2C2C;
        border-bottom: 2px solid #404040;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #B8B8B8;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        color: #1473E6;
        border-bottom: 3px solid #1473E6;
    }
    
    /* Priority badges - Rounded with glow */
    .priority-critical {
        background: linear-gradient(135deg, #D7373F 0%, #B82E35 100%);
        color: white;
        padding: 0.375rem 1rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: inline-block;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 8px rgba(215, 55, 63, 0.4);
    }
    
    .priority-high {
        background: linear-gradient(135deg, #E68619 0%, #CC7614 100%);
        color: white;
        padding: 0.375rem 1rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: inline-block;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 8px rgba(230, 134, 25, 0.4);
    }
    
    .priority-medium {
        background: linear-gradient(135deg, #1473E6 0%, #0D66D0 100%);
        color: white;
        padding: 0.375rem 1rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: inline-block;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 8px rgba(20, 115, 230, 0.4);
    }
    
    /* Recommendation cards - Smooth, rounded, elevated */
    .recommendation-card {
        background: linear-gradient(135deg, #2C2C2C 0%, #323232 100%);
        border: 1px solid #404040;
        border-radius: 12px;
        padding: 1.75rem;
        margin: 1.25rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .recommendation-card:hover {
        border-color: #1473E6;
        box-shadow: 0 6px 20px rgba(20,115,230,0.3), 0 0 0 1px rgba(20,115,230,0.2);
        transform: translateY(-3px);
    }
    
    /* ML badge */
    .ml-badge {
        background-color: #383838;
        color: #1473E6;
        padding: 0.25rem 0.5rem;
        border-radius: 3px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid #505050;
    }
    
    /* Section divider */
    hr {
        border: none;
        border-top: 1px solid #404040;
        margin: 2rem 0;
    }
    
    /* Select boxes and inputs - Rounded modern design */
    .stSelectbox > div > div, 
    .stNumberInput > div > div,
    .stSlider > div > div {
        background: linear-gradient(135deg, #2C2C2C 0%, #2A2A2A 100%);
        border-radius: 8px;
        border: 1px solid #404040;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover,
    .stNumberInput > div > div:hover {
        border-color: #1473E6;
        box-shadow: 0 0 0 2px rgba(20, 115, 230, 0.2);
    }
    
    /* Slider styling - Blended design */
    .stSlider {
        padding: 1rem 0;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #2C2C2C 0%, #2A2A2A 100%);
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Slider track */
    .stSlider > div > div > div > div {
        background: #1A1A1A !important;
        border-radius: 10px;
        height: 6px;
    }
    
    /* Slider filled track */
    .stSlider > div > div > div > div > div {
        background: linear-gradient(90deg, #1473E6 0%, #0D66D0 100%) !important;
        border-radius: 10px;
    }
    
    /* Slider thumb */
    .stSlider > div > div > div > div > div > div {
        background: linear-gradient(135deg, #1473E6 0%, #0D66D0 100%) !important;
        border: 2px solid #FFFFFF !important;
        box-shadow: 0 2px 8px rgba(20, 115, 230, 0.4) !important;
        width: 18px !important;
        height: 18px !important;
        transition: all 0.2s ease !important;
    }
    
    .stSlider > div > div > div > div > div > div:hover {
        transform: scale(1.2) !important;
        box-shadow: 0 3px 12px rgba(20, 115, 230, 0.6) !important;
    }
    
    /* Text color for general content */
    p, li, span {
        color: #D0D0D0;
    }
    
    /* Strong text */
    strong {
        color: #FFFFFF;
    }
    
    /* Links */
    a {
        color: #1473E6;
        text-decoration: none;
    }
    
    a:hover {
        color: #0D66D0;
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)


# Plotly theme for Lightroom
PLOTLY_TEMPLATE = {
    'layout': {
        'paper_bgcolor': '#242424',
        'plot_bgcolor': '#2C2C2C',
        'font': {'color': '#E8E8E8', 'family': 'Adobe Clean, Segoe UI, sans-serif'},
        'title': {'font': {'size': 18, 'color': '#FFFFFF'}},
        'xaxis': {
            'gridcolor': '#404040',
            'linecolor': '#505050',
            'zerolinecolor': '#505050'
        },
        'yaxis': {
            'gridcolor': '#404040',
            'linecolor': '#505050',
            'zerolinecolor': '#505050'
        },
        'colorway': ['#1473E6', '#268E6C', '#E68619', '#9747FF', '#D7373F', '#00B8D4']
    }
}


@st.cache_data
def load_or_generate_data():
    """Load existing data or generate new synthetic catalog."""
    try:
        catalog_df = pd.read_csv('lightroom_catalog_synthetic.csv')
        catalog_df['capture_date'] = pd.to_datetime(catalog_df['capture_date'])
        catalog_df['last_modified_date'] = pd.to_datetime(catalog_df['last_modified_date'])
        return catalog_df
    except FileNotFoundError:
        with st.spinner('Generating synthetic catalog data...'):
            generator = LightroomCatalogGenerator(num_photos=5000)
            catalog_df = generator.generate_catalog()
            catalog_df.to_csv('lightroom_catalog_synthetic.csv', index=False)
        return catalog_df


@st.cache_data
def run_analysis(catalog_df):
    """Run catalog analysis."""
    analyzer = CatalogAnalyzer(catalog_df)
    results = analyzer.run_full_analysis()
    processed_df = analyzer.get_catalog_dataframe()
    return results, processed_df


@st.cache_data
def generate_recommendations(catalog_df, analysis_results):
    """Generate ML-driven Smart Collection recommendations."""
    engine = MLRecommendationEngine(catalog_df, analysis_results)
    recommendations = engine.generate_all_recommendations()
    return recommendations


def create_gauge_chart(value, title, max_value=100):
    """Create a gauge chart for health score."""
    if value >= 76:
        color = "#268E6C"
    elif value >= 51:
        color = "#E68619"
    else:
        color = "#D7373F"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'color': '#FFFFFF'}},
        number={'font': {'size': 48, 'color': color}},
        gauge={
            'axis': {'range': [None, max_value], 'tickwidth': 1, 'tickcolor': "#808080"},
            'bar': {'color': color},
            'bgcolor': "#2C2C2C",
            'borderwidth': 2,
            'bordercolor': "#505050",
            'steps': [
                {'range': [0, 50], 'color': '#3D2020'},
                {'range': [50, 75], 'color': '#3D3220'},
                {'range': [75, 100], 'color': '#203D28'}
            ],
            'threshold': {
                'line': {'color': "#FFFFFF", 'width': 3},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="#242424",
        font={'color': "#E8E8E8", 'family': "Adobe Clean, Segoe UI, sans-serif"}
    )
    
    return fig


def page_catalog_overview():
    """Page 1: Catalog Overview with Health Score."""
    st.title("Catalog Overview")
    st.caption("Comprehensive analysis of your Lightroom catalog")
    
    # Load data
    catalog_df = load_or_generate_data()
    analysis_results, processed_df = run_analysis(catalog_df)
    
    overview = analysis_results['catalog_overview']
    health = analysis_results['health_score']
    
    # Key Stats Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("TOTAL PHOTOS", f"{overview['total_photos']:,}")
    with col2:
        st.metric("CATALOG SIZE", f"{overview['total_size_gb']:.2f} GB")
    with col3:
        date_range = f"{overview['date_range']['oldest'][:4]} - {overview['date_range']['newest'][:4]}"
        st.metric("DATE RANGE", date_range)
    with col4:
        camera_count = len(analysis_results['shooting_style']['camera_distribution'])
        st.metric("CAMERA BODIES", camera_count)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Health Score Section
    st.subheader("Organizational Health Score")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Gauge chart
        gauge = create_gauge_chart(
            health['total_health_score'],
            "Overall Health",
            max_value=100
        )
        st.plotly_chart(gauge, use_container_width=True)
        
        # Status badge
        status = health['health_category']
        if status == 'Excellent':
            st.success(f"STATUS: {status}")
        elif status == 'Good':
            st.warning(f"STATUS: {status}")
        else:
            st.error(f"STATUS: {status}")
    
    with col2:
        st.markdown("**Score Components**")
        
        # Component scores as horizontal bars
        components = [
            ("Keywords Coverage", health['keyword_score'], 30),
            ("Collection Usage", health['collection_score'], 20),
            ("Rating Consistency", health['rating_score'], 20),
            ("Folder Structure", health['folder_score'], 15),
            ("Edit Completion", health['edit_score'], 15)
        ]
        
        for name, score, max_score in components:
            pct = (score / max_score) * 100
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.progress(pct / 100)
                st.caption(name)
            with col_b:
                st.metric("", f"{score:.1f}/{max_score}")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Timeline
    st.subheader("Photo Timeline")
    
    timeline_df = processed_df.copy()
    timeline_df['year_month'] = pd.to_datetime(timeline_df['capture_date']).dt.to_period('M')
    monthly_counts = timeline_df.groupby('year_month').size().reset_index(name='count')
    monthly_counts['year_month'] = monthly_counts['year_month'].astype(str)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly_counts['year_month'],
        y=monthly_counts['count'],
        marker_color='#1473E6',
        hovertemplate='%{x}<br>Photos: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title='Photos Captured per Month',
        xaxis_title='Month',
        yaxis_title='Number of Photos',
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # File Type and Camera Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("File Types")
        file_dist = pd.DataFrame.from_dict(
            overview['file_type_distribution'],
            orient='index',
            columns=['Count']
        ).reset_index()
        file_dist.columns = ['File Type', 'Count']
        
        fig = go.Figure(data=[go.Pie(
            labels=file_dist['File Type'],
            values=file_dist['Count'],
            hole=0.4,
            marker=dict(colors=['#1473E6', '#268E6C', '#E68619', '#9747FF']),
            textfont=dict(color='#FFFFFF')
        )])
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            title='File Type Distribution',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Camera Bodies")
        camera_dist = pd.DataFrame.from_dict(
            analysis_results['shooting_style']['camera_distribution'],
            orient='index',
            columns=['Count']
        ).reset_index()
        camera_dist.columns = ['Camera', 'Count']
        camera_dist = camera_dist.sort_values('Count', ascending=True)
        
        fig = go.Figure(go.Bar(
            x=camera_dist['Count'],
            y=camera_dist['Camera'],
            orientation='h',
            marker_color='#1473E6',
            hovertemplate='%{y}<br>Photos: %{x}<extra></extra>'
        ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            title='Photos by Camera Body',
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)


def page_recommendations():
    """Page 2: Smart Collection Recommendations."""
    st.title("Smart Collection Recommendations")
    st.caption("Personalized suggestions based on your catalog patterns")
    
    catalog_df = load_or_generate_data()
    analysis_results, processed_df = run_analysis(catalog_df)
    recommendations = generate_recommendations(processed_df, analysis_results)
    
    # Filter controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        categories = ['All Categories'] + sorted(list(set([r.category for r in recommendations])))
        selected_category = st.selectbox("FILTER BY CATEGORY", categories)
    
    with col2:
        min_priority = st.slider("MIN PRIORITY", 0, 100, 0)
    
    with col3:
        show_count = st.number_input("SHOW TOP", 1, len(recommendations), min(10, len(recommendations)))
    
    # Filter recommendations
    filtered_recs = recommendations
    if selected_category != 'All Categories':
        filtered_recs = [r for r in filtered_recs if r.category == selected_category]
    filtered_recs = [r for r in filtered_recs if r.priority_score >= min_priority]
    filtered_recs = filtered_recs[:show_count]
    
    st.markdown(f"**Showing {len(filtered_recs)} recommendations**")
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Display recommendations
    for i, rec in enumerate(filtered_recs, 1):
        # Priority badge HTML
        if rec.priority_score >= 80:
            priority_badge = '<div class="priority-critical">CRITICAL PRIORITY</div>'
        elif rec.priority_score >= 60:
            priority_badge = '<div class="priority-high">HIGH PRIORITY</div>'
        else:
            priority_badge = '<div class="priority-medium">MEDIUM PRIORITY</div>'
        
        # Recommendation card
        st.markdown(f"""
        <div class="recommendation-card">
            {priority_badge}
            <h3 style="margin-top: 0.5rem;">{i}. {rec.collection_name}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Category:** {rec.category}")
            st.markdown(f"**Impact:** {rec.impact_description}")
            st.markdown(f"**Why Recommended:** {rec.why_recommended}")
            st.markdown(f"**Expected Benefit:** {rec.expected_benefit}")
        
        with col2:
            st.metric("PHOTOS AFFECTED", f"{rec.photos_affected:,}")
            st.metric("PRIORITY", f"{rec.priority_score:.0f}/100")
            
            # Percentage bar
            pct = (rec.photos_affected / len(catalog_df)) * 100
            st.progress(min(pct / 100, 1.0))
            st.caption(f"{pct:.1f}% of catalog")
        
        # Setup instructions
        with st.expander("IMPLEMENTATION GUIDE"):
            st.markdown(f"**Lightroom Smart Collection Rule:**")
            st.code(rec.lightroom_rule_syntax, language="text")
            st.caption(f"Copy this rule and paste in: Library > Smart Collection > Add Rule")
            
            if st.button(f"MARK AS IMPLEMENTED", key=f"done_{rec.recommendation_id}"):
                st.success("Great! This collection will help organize your catalog.")
        
        st.markdown("<hr>", unsafe_allow_html=True)


def main():
    """Main application logic."""
    
    # Sidebar
    st.sidebar.image("Lightroom.png", width=60)
    st.sidebar.title("Smart Collections Intelligence")
    st.sidebar.caption("Catalog Analysis & Recommendations")
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "NAVIGATE",
        [
            "Catalog Overview",
            "Recommendations"
        ],
        label_visibility="visible"
    )
    
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.caption("**About**")
    st.sidebar.caption("Analyzes your Lightroom catalog to identify organizational patterns and suggest Smart Collections that improve workflow efficiency.")
    
    # Route to pages
    if page == "Catalog Overview":
        page_catalog_overview()
    elif page == "Recommendations":
        page_recommendations()


if __name__ == "__main__":
    main()