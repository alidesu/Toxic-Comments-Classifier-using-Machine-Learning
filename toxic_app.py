import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import re
import os

# Try to import plotly, use fallback if not available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("📊 Plotly not installed. Some visualizations will be simplified.")

# Page configuration
st.set_page_config(
    page_title="Toxic Comment Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .toxic-alert {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #f44336;
    }
    .safe-alert {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Load model and metadata
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        return model, vectorizer, metadata
    except FileNotFoundError:
        st.error("🚨 Model files not found. Please run the training notebook first!")
        return None, None, None

# Text preprocessing (same as training)
def clean_text(text):
    if pd.isna(text) or text.strip() == "":
        return ""
    
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Header
st.markdown('<div class="main-header">🔍 Toxic Comment Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered content moderation system</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 Model Information")
    
    model, vectorizer, metadata = load_model()
    
    if metadata:
        st.markdown("### Model Performance")
        st.metric("Best Model", metadata.get('best_model', 'Unknown'))
        st.metric("Accuracy", f"{metadata.get('accuracy', 0):.1%}")
        st.metric("AUC Score", f"{metadata.get('auc_score', 0):.3f}")
        st.metric("Training Samples", f"{metadata.get('training_samples', 0):,}")
        
        st.markdown("### Model Features")
        st.info(f"📝 Vocabulary Size: {metadata.get('vocab_size', 0):,}")
        st.info(f"🔤 Features: {metadata.get('vectorizer_type', 'TF-IDF')}")
        st.info(f"📅 Trained: {metadata.get('training_date', 'Unknown')}")
    
    st.markdown("### About")
    st.markdown("""
    This classifier uses machine learning to detect toxic comments.
    
    **Features:**
    - Multiple model comparison
    - Real-time prediction
    - Confidence scoring
    - Text preprocessing
    """)

# Main content
if model is not None and vectorizer is not None:
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📊 Batch Analysis", "📈 Analytics"])
    
    with tab1:
        st.markdown("### Analyze a Comment")
        
        # Input methods
        input_method = st.radio("Choose input method:", ["Text Area", "File Upload"], horizontal=True)
        
        if input_method == "Text Area":
            comment = st.text_area(
                "Enter comment to analyze:",
                placeholder="Type or paste your comment here...",
                height=150
            )
        else:
            uploaded_file = st.file_uploader("Upload a text file", type=['txt'])
            comment = ""
            if uploaded_file:
                comment = str(uploaded_file.read(), "utf-8")
                st.text_area("File content:", comment, height=150, disabled=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            analyze_button = st.button("🔍 Analyze Comment", use_container_width=True, type="primary")
        
        if analyze_button and comment.strip():
            # Clean and process text
            cleaned_comment = clean_text(comment)
            
            if len(cleaned_comment.split()) < 3:
                st.warning("⚠️ Comment too short or empty after cleaning. Please enter a longer comment.")
            else:
                # Make prediction
                vec = vectorizer.transform([cleaned_comment])
                pred = model.predict(vec)[0]
                prob = model.predict_proba(vec)[0]
                
                # Store in history
                st.session_state.history.append({
                    'timestamp': datetime.now(),
                    'comment': comment[:100] + "..." if len(comment) > 100 else comment,
                    'prediction': 'Toxic' if pred == 1 else 'Safe',
                    'confidence': max(prob)
                })
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    if pred == 1:
                        st.markdown("""
                        <div class="toxic-alert">
                            <h3>🚨 Toxic Content Detected</h3>
                            <p>This comment appears to contain toxic content.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="safe-alert">
                            <h3>✅ Content Appears Safe</h3>
                            <p>This comment appears to be non-toxic.</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    # Confidence visualization
                    if PLOTLY_AVAILABLE:
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = prob[1] * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Toxicity Confidence"},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkred" if pred == 1 else "darkgreen"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgreen"},
                                    {'range': [50, 80], 'color': "yellow"},
                                    {'range': [80, 100], 'color': "lightcoral"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Fallback visualization using Streamlit's built-in progress bar
                        st.markdown("### Toxicity Confidence")
                        confidence_score = prob[1] * 100
                        st.progress(confidence_score / 100)
                        st.metric("Confidence Score", f"{confidence_score:.1f}%")
                
                # Detailed scores
                st.markdown("### Detailed Scores")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Non-Toxic Probability", f"{prob[0]:.1%}")
                with col2:
                    st.metric("Toxic Probability", f"{prob[1]:.1%}")
        
        elif analyze_button:
            st.warning("Please enter a comment to analyze.")
    
    with tab2:
        st.markdown("### Batch Analysis")
        st.info("Upload a CSV file with a 'comment' column for batch analysis.")
        
        uploaded_csv = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_csv:
            try:
                df = pd.read_csv(uploaded_csv)
                
                if 'comment' not in df.columns:
                    st.error("CSV must contain a 'comment' column.")
                else:
                    st.success(f"Loaded {len(df)} comments.")
                    
                    if st.button("Analyze All Comments"):
                        progress_bar = st.progress(0)
                        results = []
                        
                        for i, comment in enumerate(df['comment']):
                            cleaned = clean_text(str(comment))
                            if len(cleaned.split()) >= 3:
                                vec = vectorizer.transform([cleaned])
                                pred = model.predict(vec)[0]
                                prob = model.predict_proba(vec)[0]
                                
                                results.append({
                                    'comment': comment,
                                    'prediction': 'Toxic' if pred == 1 else 'Safe',
                                    'toxic_probability': prob[1],
                                    'safe_probability': prob[0]
                                })
                            
                            progress_bar.progress((i + 1) / len(df))
                        
                        results_df = pd.DataFrame(results)
                        
                        # Summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Analyzed", len(results_df))
                        with col2:
                            toxic_count = len(results_df[results_df['prediction'] == 'Toxic'])
                            st.metric("Toxic Comments", toxic_count)
                        with col3:
                            st.metric("Toxic Rate", f"{toxic_count/len(results_df)*100:.1f}%")
                        
                        # Visualization
                        if PLOTLY_AVAILABLE:
                            fig = px.histogram(results_df, x='prediction', color='prediction',
                                             title="Batch Analysis Results")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Fallback using Streamlit's built-in bar chart
                            st.markdown("### Batch Analysis Results")
                            chart_data = results_df['prediction'].value_counts()
                            st.bar_chart(chart_data)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="toxicity_analysis_results.csv",
                            mime="text/csv"
                        )
                        
                        st.dataframe(results_df)
                        
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    with tab3:
        st.markdown("### Usage Analytics")
        
        if st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Predictions", len(history_df))
            with col2:
                toxic_rate = len(history_df[history_df['prediction'] == 'Toxic']) / len(history_df)
                st.metric("Toxic Rate", f"{toxic_rate:.1%}")
            with col3:
                avg_confidence = history_df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            
            # Timeline
            if PLOTLY_AVAILABLE:
                fig = px.scatter(history_df, x='timestamp', y='confidence', 
                               color='prediction', title="Prediction Timeline")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback using Streamlit's built-in line chart
                st.markdown("### Prediction Timeline")
                timeline_data = history_df.set_index('timestamp')[['confidence']]
                st.line_chart(timeline_data)
            
            # Recent history
            st.markdown("### Recent Predictions")
            recent = history_df.tail(10).sort_values('timestamp', ascending=False)
            st.dataframe(recent[['timestamp', 'comment', 'prediction', 'confidence']])
            
            if st.button("Clear History"):
                st.session_state.history = []
                st.rerun()
        else:
            st.info("No predictions made yet. Try analyzing some comments!")

else:
    st.error("🚨 Model not loaded. Please run the training notebook first to create the model files.")
    st.markdown("""
    ### Setup Instructions:
    1. Run all cells in the Jupyter notebook
    2. Ensure model.pkl, vectorizer.pkl, and model_metadata.pkl are created
    3. Restart this Streamlit app
    """)
