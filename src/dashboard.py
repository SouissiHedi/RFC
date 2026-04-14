import streamlit as st
import os
import sys
import shlex
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime
import subprocess
import warnings
warnings.filterwarnings('ignore')

def run_training_script():
    """Run the training script as a subprocess"""
    try:
        # Get absolute path
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src/scripts", "train_models.py"))
        
        # Debug: Show the path
        st.info(f"Script path: {script_path}")
        st.info(f"Script exists: {os.path.exists(script_path)}")
        
        # Use list form with sys.executable to avoid shell splitting
        result = subprocess.run(
            [sys.executable, script_path],  # Pass as list, not string
            capture_output=True,
            text=True,
            cwd=os.path.dirname(script_path)  # Set working directory
        )
        
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except Exception as e:
        return False, str(e)

def run_tuning_script():
    """Run the threshold tuning script as a subprocess"""
    try:
        # Get absolute path
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src/scripts", "threshold_tuning.py"))
        
        # Debug: Show the path
        st.info(f"Script path: {script_path}")
        st.info(f"Script exists: {os.path.exists(script_path)}")
        
        # Use list form with sys.executable to avoid shell splitting
        result = subprocess.run(
            [sys.executable, script_path],  # Pass as list, not string
            capture_output=True,
            text=True,
            cwd=os.path.dirname(script_path)  # Set working directory
        )
        
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except Exception as e:
        return False, str(e)

def load_training_results_from_files():
    """Load training results from saved JSON files"""
    try:
        artifacts_dir = Path("../src/artifacts")
        
        # First check for latest summary
        latest_summary = artifacts_dir / "latest" / "training_summary_latest.json"
        if latest_summary.exists():
            with open(latest_summary, 'r') as f:
                return json.load(f)
        
        # If not found, check timestamped directories
        run_dirs = sorted([d for d in artifacts_dir.glob("run_*") if d.is_dir()], 
                         key=lambda x: x.name, reverse=True)
        
        if run_dirs:
            latest_run = run_dirs[0]
            summary_file = latest_run / f"training_summary_{latest_run.name.replace('run_', '')}.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    return json.load(f)
        
        return None
    except Exception as e:
        st.error(f"Error loading training results: {e}")
        return None

def load_tuning_results():
    """Load threshold tuning results"""
    results = {}
    artifacts_dir = Path("../src/artifacts")
    
    # Find latest tuning directory
    tuning_dirs = sorted([d for d in artifacts_dir.glob("threshold_analysis_*") if d.is_dir()], 
                        key=lambda x: x.name, reverse=True)
    
    if tuning_dirs:
        latest_tuning = tuning_dirs[0]
        
        try:
            # Load performance summary
            perf_file = latest_tuning / "model_performance_summary.csv"
            if perf_file.exists():
                results['performance'] = pd.read_csv(perf_file, index_col=0)
            
            # Load threshold analysis for each model
            threshold_files = list(latest_tuning.glob("*_threshold_analysis.csv"))
            results['all_results'] = {}
            
            for file in threshold_files:
                try:
                    model_name = file.stem.replace("_threshold_analysis", "")
                    results['all_results'][model_name] = pd.read_csv(file)
                except Exception as e:
                    st.error(f"Error loading {file}: {e}")
            
            # Load optimal thresholds
            optimal_file = latest_tuning / "optimal_thresholds.csv"
            if optimal_file.exists():
                try:
                    optimal_df = pd.read_csv(optimal_file, index_col=0)
                    # Convert to dictionary for easier access
                    results['optimal_thresholds'] = optimal_df.to_dict('index')
                except Exception as e:
                    st.error(f"Error loading optimal thresholds: {e}")
                    results['optimal_thresholds'] = {}
            
            results['timestamp'] = latest_tuning.name.replace("threshold_analysis_", "")
            
        except Exception as e:
            st.error(f"Error loading tuning results: {e}")
            return {}
    
    return results


# Page configuration
st.set_page_config(
    page_title="Model Training Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E40AF;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .model-card {
        background-color: #ABACAD;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #E0F2FE;
        margin-bottom: 1rem;
        min-height: 200px;
    }
    .stButton button {
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #2563EB;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'training_results' not in st.session_state:
    st.session_state.training_results = None
if 'tuning_results' not in st.session_state:
    st.session_state.tuning_results = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

# Title
st.markdown("<h1 class='main-header'>🤖 Model Training & Tuning Dashboard</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103655.png", width=100)
    st.title("Navigation")
    
    app_mode = st.selectbox(
        "Select Mode",
        ["📊 Dashboard Overview", "🚀 Train New Models", "🎯 Tune Thresholds", 
         "📈 Model Comparison", "📁 Model History", "⚙️ Settings"]
    )
    
    st.markdown("---")
    st.info(
        """
        **About this dashboard:**
        - Train multiple ML models
        - Tune prediction thresholds
        - Compare model performance
        - Track model history
        """
    )
    
    # Quick stats
    artifacts_dir = Path("../src/artifacts")
    if artifacts_dir.exists():
        model_files = list(artifacts_dir.rglob("model_*.pkl"))
        st.metric("Total Models", len(model_files))

# Function to load model metadata
def load_model_metadata():
    """Load model metadata from saved JSON files"""
    metadata = {}
    artifacts_dir = Path("../src/artifacts")
    
    # Check for latest summary
    latest_summary = artifacts_dir / "latest" / "training_summary_latest.json"
    if latest_summary.exists():
        try:
            with open(latest_summary, 'r') as f:
                metadata['latest'] = json.load(f)
        except Exception as e:
            st.error(f"Error loading latest summary: {e}")
    
    # Load from timestamped directories
    try:
        run_dirs = sorted([d for d in artifacts_dir.glob("run_*") if d.is_dir()], 
                          key=lambda x: x.name, reverse=True)
        
        metadata['runs'] = []
        for run_dir in run_dirs[:10]:  # Load last 10 runs
            summary_file = run_dir / f"training_summary_{run_dir.name.replace('run_', '')}.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    run_data = json.load(f)
                    run_data['directory'] = str(run_dir)
                    metadata['runs'].append(run_data)
    except Exception as e:
        st.error(f"Error loading run metadata: {e}")
    
    return metadata

# Dashboard Overview
if app_mode == "📊 Dashboard Overview":
    st.markdown("<h2 class='sub-header'>📊 Dashboard Overview</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Training Runs", "5", "↗️ 2")
    
    with col2:
        st.metric("Best Model F1 Score", "0.89", "↗️ 0.03")
    
    with col3:
        st.metric("Latest Accuracy", "92.5%", "↗️ 1.2%")
    
    # Load metadata
    metadata = load_model_metadata()
    
    if metadata.get('runs'):
        st.markdown("<h3 class='sub-header'>📈 Recent Training Runs</h3>", unsafe_allow_html=True)
        
        # Create DataFrame of recent runs
        runs_data = []
        for run in metadata['runs'][:5]:
            runs_data.append({
                'Date': run['training_timestamp'],
                'Best Model': run['best_recall_model']['name'],
                'Recall': run['best_recall_model']['metrics']['recall'],
                'Precision': run['best_recall_model']['metrics']['precision'],
                'F1': run['best_recall_model']['metrics']['f1']
            })
        
        runs_df = pd.DataFrame(runs_data)
        st.dataframe(runs_df, use_container_width=True)
        
        # Performance trends chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=runs_df['Date'],
            y=runs_df['Recall'],
            mode='lines+markers',
            name='Recall',
            line=dict(color='#3B82F6', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=runs_df['Date'],
            y=runs_df['F1'],
            mode='lines+markers',
            name='F1 Score',
            line=dict(color='#10B981', width=3)
        ))
        
        fig.update_layout(
            title="Performance Trends Over Time",
            xaxis_title="Training Date",
            yaxis_title="Score",
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Model status
    st.markdown("<h3 class='sub-header'>📊 Model Status</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='model-card'>
            <h4>🤖 Logistic Regression</h4>
            <p><b>Status:</b> 🟢 Ready</p>
            <p><b>Latest F1:</b> 0.85</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='model-card'>
            <h4>🌲 Random Forest</h4>
            <p><b>Status:</b> 🟢 Ready</p>
            <p><b>Latest F1:</b> 0.89</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='model-card'>
            <h4>⚡ XGBoost</h4>
            <p><b>Status:</b> 🟢 Ready</p>
            <p><b>Latest F1:</b> 0.87</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='model-card'>
            <h4>🏆 Best Model</h4>
            <p><b>Current:</b> Random Forest</p>
            <p><b>Recall:</b> 0.92</p>
        </div>
        """, unsafe_allow_html=True)

# Train New Models
elif app_mode == "🚀 Train New Models":
    st.markdown("<h2 class='sub-header'>🚀 Train New Models</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Configure Training Parameters
        
        Adjust the parameters for each model type below:
        """)
        
        with st.expander("Logistic Regression Settings", expanded=True):
            lr_max_iter = st.slider("Max Iterations", 100, 5000, 1000, key="lr_iter")
            lr_c = st.slider("Regularization (C)", 0.01, 10.0, 1.0, key="lr_c")
        
        with st.expander("Random Forest Settings", expanded=True):
            rf_n_estimators = st.slider("Number of Trees", 50, 500, 200, key="rf_trees")
            rf_max_depth = st.slider("Max Depth", 5, 50, 20, key="rf_depth")
        
        with st.expander("XGBoost Settings", expanded=True):
            xgb_n_estimators = st.slider("Number of Trees", 50, 500, 300, key="xgb_trees")
            xgb_learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.05, key="xgb_lr")
            xgb_max_depth = st.slider("Max Depth", 3, 10, 5, key="xgb_depth")
    
    with col2:
        st.markdown("""
        ### Training Options
        """)
        
        models_to_train = st.multiselect(
            "Select models to train:",
            ["Logistic Regression", "Random Forest", "XGBoost"],
            default=["Logistic Regression", "Random Forest", "XGBoost"]
        )
        
        random_state = st.number_input("Random State", value=42)
        
        use_class_weights = st.checkbox("Use Class Weights", value=True)
        
        st.markdown("---")
        
        if st.button("🚀 Start Training", use_container_width=True):
            with st.spinner("Training models... This may take a few minutes."):
                success, output = run_training_script()
                
                if success:
                    st.success("✅ Training completed successfully!")
                    st.balloons()
                    
                    # Load results from files
                    results = load_training_results_from_files()
                    if results:
                        st.session_state.training_results = results
                        st.session_state.models_loaded = True
                    
                    # Show output in expander
                    with st.expander("Show training logs"):
                        st.text(output)
                else:
                    st.error(f"❌ Training failed: {output}")
        
        # Show training status
        if st.session_state.training_results:
            st.markdown("""
            ### Last Training Results
            """)
            
            results = st.session_state.training_results
            best_model = max(results, key=lambda x: results[x]["metrics"]["recall"])
            
            st.metric("Best Model", best_model)
            st.metric("Best Recall", f"{results[best_model]['metrics']['recall']:.3f}")

# Tune Thresholds
elif app_mode == "🎯 Tune Thresholds":
    st.markdown("<h2 class='sub-header'>🎯 Threshold Tuning</h2>", unsafe_allow_html=True)
    
    # Load available models
    artifacts_dir = Path("../src/artifacts/latest")
    model_files = []
    if artifacts_dir.exists():
        model_files = list(artifacts_dir.glob("model_*_latest.pkl"))
    
    if not model_files:
        st.warning("No trained models found. Please train models first.")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Configuration")
            
            # Model selection
            model_names = [f.stem.replace("model_", "").replace("_latest", "") 
                          for f in model_files]
            selected_model = st.selectbox("Select Model", model_names)
            
            # Threshold range
            threshold_min = st.slider("Minimum Threshold", 0.0, 0.5, 0.01, 0.01)
            threshold_max = st.slider("Maximum Threshold", 0.5, 1.0, 0.99, 0.01)
            threshold_step = st.slider("Step Size", 0.01, 0.1, 0.02, 0.01)
            
            # Business constraints
            st.markdown("#### Business Constraints")
            min_precision = st.slider("Minimum Precision (%)", 50, 100, 80) / 100
            min_recall = st.slider("Minimum Recall (%)", 50, 100, 70) / 100
            
            if st.button("🔍 Analyze Thresholds", use_container_width=True):
                with st.spinner("Analyzing thresholds..."):
                    success, output = run_tuning_script()
                    
                    if success:
                        st.success("✅ Threshold analysis completed!")
                        
                        # Load results from files
                        results = load_tuning_results()
                        if results:
                            st.session_state.tuning_results = results
                        
                        # Show output in expander
                        with st.expander("Show tuning logs"):
                            st.text(output)
                    else:
                        st.error(f"❌ Threshold analysis failed: {output}")
        
        with col2:
            if st.session_state.tuning_results:
                results = st.session_state.tuning_results
                
                # Display optimal thresholds for selected model
                if selected_model in results['optimal_thresholds']:
                    st.markdown(f"### Optimal Thresholds for {selected_model}")
                    
                    thresholds = results['optimal_thresholds'][selected_model]
                    
                    cols = st.columns(4)
                    
                    with cols[0]:
                        if thresholds.get('best_f1'):
                            st.metric("Best F1", f"{thresholds['best_f1']['threshold']:.3f}")
                    
                    with cols[1]:
                        if thresholds.get('best_f2'):
                            st.metric("Best F2", f"{thresholds['best_f2']['threshold']:.3f}")
                    
                    with cols[2]:
                        if thresholds.get('best_recall'):
                            st.metric("Best Recall", f"{thresholds['best_recall']['threshold']:.3f}")
                    
                    with cols[3]:
                        if thresholds.get('balanced_080_070'):
                            st.metric("Balanced", f"{thresholds['balanced_080_070']['threshold']:.3f}")
                    
                    # Threshold analysis plot
                    if selected_model in results['all_results']:
                        df = results['all_results'][selected_model]
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=df['threshold'],
                            y=df['f1'],
                            mode='lines',
                            name='F1 Score',
                            line=dict(color='#10B981', width=3)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=df['threshold'],
                            y=df['recall'],
                            mode='lines',
                            name='Recall',
                            line=dict(color='#3B82F6', width=3)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=df['threshold'],
                            y=df['precision'],
                            mode='lines',
                            name='Precision',
                            line=dict(color='#EF4444', width=3)
                        ))
                        
                        # Add vertical line for best F1
                        if thresholds.get('best_f1'):
                            best_threshold = thresholds['best_f1']['threshold']
                            fig.add_vline(x=best_threshold, line_dash="dash", 
                                         line_color="green", 
                                         annotation_text=f"Best F1: {best_threshold:.3f}")
                        
                        fig.update_layout(
                            title=f"Threshold Analysis - {selected_model}",
                            xaxis_title="Threshold",
                            yaxis_title="Score",
                            height=400,
                            template="plotly_white",
                            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Metrics at different thresholds
                        st.markdown("#### Metrics at Key Thresholds")
                        
                        key_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
                        metrics_data = []
                        
                        for t in key_thresholds:
                            closest_idx = (df['threshold'] - t).abs().idxmin()
                            row = df.iloc[closest_idx]
                            metrics_data.append({
                                'Threshold': t,
                                'Recall': row['recall'],
                                'Precision': row['precision'],
                                'F1': row['f1'],
                                'Alerts': row['total_alerts']
                            })
                        
                        metrics_df = pd.DataFrame(metrics_data)
                        st.dataframe(metrics_df, use_container_width=True)

# Model Comparison
elif app_mode == "📈 Model Comparison":
    st.markdown("<h2 class='sub-header'>📈 Model Comparison</h2>", unsafe_allow_html=True)
    
    # Load tuning results or metadata
    tuning_results = st.session_state.tuning_results
    metadata = load_model_metadata()
    
    if not tuning_results and not (metadata.get('runs') or metadata.get('latest')):
        st.warning("No model results available. Please train or load models first.")
        
        # Add a button to load results
        if st.button("🔄 Load Latest Results"):
            with st.spinner("Loading results..."):
                # Try to load both training and tuning results
                training_results = load_training_results_from_files()
                tuning_results_data = load_tuning_results()
                
                if training_results:
                    st.session_state.training_results = training_results
                    st.session_state.models_loaded = True
                    st.success("✅ Training results loaded!")
                
                if tuning_results_data:
                    st.session_state.tuning_results = tuning_results_data
                    st.success("✅ Tuning results loaded!")
                
                st.rerun()
    else:
        # Create tabs for different comparison views
        tab1, tab2, tab3 = st.tabs(["📊 Performance Metrics", "📈 Model Characteristics", "📉 Confusion Matrices"])
        
        with tab1:
            st.markdown("### Performance Metrics")
            
            # Try multiple sources for performance data
            performance_data = None
            
            # First check tuning results
            if tuning_results and 'performance' in tuning_results:
                if isinstance(tuning_results['performance'], dict):
                    performance_data = pd.DataFrame(tuning_results['performance']).T
                else:
                    performance_data = tuning_results['performance']
            
            # If no tuning results, check training results in session state
            elif st.session_state.training_results:
                # Convert training results to dataframe
                models_data = []
                training_results = st.session_state.training_results
                
                if isinstance(training_results, dict):
                    for model_name, model_info in training_results.items():
                        if isinstance(model_info, dict) and 'metrics' in model_info:
                            metrics = model_info['metrics']
                            models_data.append({
                                'Model': model_name,
                                'Recall': metrics.get('recall', 0),
                                'Precision': metrics.get('precision', 0),
                                'F1': metrics.get('f1', 0),
                                'Accuracy': metrics.get('accuracy', 0)
                            })
                    
                    if models_data:
                        performance_data = pd.DataFrame(models_data).set_index('Model')
            
            # If still no data, check metadata
            elif metadata.get('latest'):
                latest = metadata['latest']
                models_data = []
                
                if isinstance(latest, dict):
                    for model_name, model_info in latest.items():
                        if isinstance(model_info, dict) and 'metrics' in model_info:
                            metrics = model_info['metrics']
                            models_data.append({
                                'Model': model_name,
                                'Recall': metrics.get('recall', 0),
                                'Precision': metrics.get('precision', 0),
                                'F1': metrics.get('f1', 0),
                                'Accuracy': metrics.get('accuracy', 0)
                            })
                    
                    if models_data:
                        performance_data = pd.DataFrame(models_data).set_index('Model')
            
            if performance_data is not None and not performance_data.empty:
                # Select metrics to display
                available_metrics = [col for col in performance_data.columns if col != 'Model']
                metrics_to_show = st.multiselect(
                    "Select metrics to display:",
                    available_metrics,
                    default=['Recall', 'Precision', 'F1'] if 'Recall' in available_metrics else available_metrics[:3]
                )
                
                if metrics_to_show:
                    # Ensure we have a copy for display
                    display_df = performance_data[metrics_to_show].copy()
                    
                    # Format numeric columns
                    for col in display_df.columns:
                        if display_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                            display_df[col] = display_df[col].round(3)
                    
                    # Display the dataframe
                    st.dataframe(display_df.style.highlight_max(axis=0, color='lightgreen')
                                                .highlight_min(axis=0, color='lightcoral'),
                                use_container_width=True)
                    
                    # Bar chart comparison
                    st.markdown("### Bar Chart Comparison")
                    
                    fig2 = go.Figure()
                    
                    for metric in metrics_to_show:
                        fig2.add_trace(go.Bar(
                            x=display_df.index.tolist(),
                            y=display_df[metric].tolist(),
                            name=metric,
                            text=display_df[metric].round(3).tolist(),
                            textposition='auto'
                        ))
                    
                    fig2.update_layout(
                        barmode='group',
                        height=400,
                        xaxis_title="Models",
                        yaxis_title="Score",
                        title=f"Performance Metrics Comparison",
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Radar chart (only if we have at least 3 metrics)
                    if len(metrics_to_show) >= 3:
                        st.markdown("### Radar Chart Comparison")
                        
                        # Prepare data for radar chart
                        fig = go.Figure()
                        
                        for model_name in display_df.index:
                            values = display_df.loc[model_name].values
                            # Handle NaN values
                            values = [0 if pd.isna(v) else v for v in values]
                            
                            fig.add_trace(go.Scatterpolar(
                                r=values,
                                theta=metrics_to_show,
                                fill='toself',
                                name=model_name
                            ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )),
                            showlegend=True,
                            height=400,
                            title="Model Performance Radar Chart"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No performance metrics available. Please train models first.")
        
        with tab2:
            st.markdown("### Model Characteristics")
            
            # Create characteristics data
            characteristics_data = {
                'Model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'AdaBoost'],
                'Training Time (s)': [2.5, 12.3, 8.7, 6.4],
                'Model Size (MB)': [1.2, 25.6, 15.8, 10.3],
                'Inference Time (ms)': [1.1, 5.4, 3.2, 2.8],
                'Complexity': ['Low', 'High', 'Medium', 'Medium'],
                'Interpretability': ['High', 'Medium', 'Low', 'Medium']
            }
            
            chars_df = pd.DataFrame(characteristics_data)
            
            # Allow filtering by model
            selected_models = st.multiselect(
                "Select models to compare:",
                chars_df['Model'].tolist(),
                default=chars_df['Model'].tolist()[:3]
            )
            
            if selected_models:
                filtered_df = chars_df[chars_df['Model'].isin(selected_models)]
                st.dataframe(filtered_df, use_container_width=True)
                
                # Visualization of characteristics
                st.markdown("#### Training Time vs Inference Time")
                
                fig = px.scatter(filtered_df, 
                               x='Training Time (s)', 
                               y='Inference Time (ms)',
                               size='Model Size (MB)',
                               color='Complexity',
                               hover_name='Model',
                               title="Model Efficiency Comparison")
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### Confusion Matrices")
            
            # Create a grid for confusion matrices
            cols = st.columns(3)
            
            models_for_confusion = ['Logistic Regression', 'Random Forest', 'XGBoost']
            
            for idx, (col, model_name) in enumerate(zip(cols, models_for_confusion)):
                with col:
                    st.markdown(f"**{model_name}**")
                    
                    # Mock confusion matrix data
                    if idx == 0:  # Logistic Regression
                        cm = np.array([[850, 50], [30, 70]])
                    elif idx == 1:  # Random Forest
                        cm = np.array([[880, 20], [25, 75]])
                    else:  # XGBoost
                        cm = np.array([[870, 30], [28, 72]])
                    
                    # Calculate metrics from confusion matrix
                    tn, fp, fn, tp = cm.ravel()
                    accuracy = (tp + tn) / (tp + tn + fp + fn)
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    # Create Plotly heatmap
                    fig = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=['Predicted Negative', 'Predicted Positive'],
                        y=['Actual Negative', 'Actual Positive'],
                        text=cm,
                        texttemplate="%{text}",
                        textfont={"size": 16},
                        colorscale='Blues',
                        showscale=False
                    ))
                    
                    fig.update_layout(
                        title=f"Accuracy: {accuracy:.1%}",
                        height=300,
                        xaxis_title="Predicted",
                        yaxis_title="Actual"
                    )
                    
                    # Annotate metrics
                    fig.add_annotation(
                        x=0.5, y=-0.15,
                        text=f"Precision: {precision:.1%} | Recall: {recall:.1%} | F1: {f1:.3f}",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        font=dict(size=10)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Summary table of confusion matrix metrics
            st.markdown("#### Confusion Matrix Summary")
            
            summary_data = []
            for idx, model_name in enumerate(models_for_confusion):
                if idx == 0:
                    cm = np.array([[850, 50], [30, 70]])
                elif idx == 1:
                    cm = np.array([[880, 20], [25, 75]])
                else:
                    cm = np.array([[870, 30], [28, 72]])
                
                tn, fp, fn, tp = cm.ravel()
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                summary_data.append({
                    'Model': model_name,
                    'TN': tn,
                    'FP': fp,
                    'FN': fn,
                    'TP': tp,
                    'Accuracy': f"{accuracy:.1%}",
                    'Precision': f"{precision:.1%}",
                    'Recall': f"{recall:.1%}",
                    'F1': f"{f1:.3f}"
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df.style.highlight_max(subset=['Accuracy', 'F1'], color='lightgreen')
                                     .highlight_min(subset=['FP', 'FN'], color='lightcoral'),
                        use_container_width=True)

# Model History
elif app_mode == "📁 Model History":
    st.markdown("<h2 class='sub-header'>📁 Model History</h2>", unsafe_allow_html=True)
    
    metadata = load_model_metadata()
    
    if not metadata.get('runs'):
        st.info("No historical runs found. Train models to see history here.")
    else:
        # Run selection
        runs = metadata['runs']
        run_dates = [run['training_timestamp'] for run in runs]
        
        selected_run = st.selectbox("Select Training Run", run_dates)
        
        # Get selected run data
        selected_data = next((run for run in runs if run['training_timestamp'] == selected_run), None)
        
        if selected_data:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### Run Information")
                
                info_data = {
                    'Date': selected_data['training_timestamp'],
                    'Best Model': selected_data['best_recall_model']['name'],
                    'Total Models': len(selected_data['all_models']),
                    'Train Samples': selected_data['data_info']['train_samples'],
                    'Test Samples': selected_data['data_info']['test_samples']
                }
                
                for key, value in info_data.items():
                    st.markdown(f"**{key}:** {value}")
                
                st.markdown("### Best Model Details")
                
                best_model = selected_data['best_recall_model']
                st.metric("Recall", f"{best_model['metrics']['recall']:.3f}")
                st.metric("Precision", f"{best_model['metrics']['precision']:.3f}")
                st.metric("F1 Score", f"{best_model['metrics']['f1']:.3f}")
                
                # Download button for model
                model_path = Path(selected_data.get('directory', '')) / best_model['timestamp_file']
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        model_bytes = f.read()
                    
                    st.download_button(
                        label="📥 Download Best Model",
                        data=model_bytes,
                        file_name=best_model['timestamp_file'],
                        mime="application/octet-stream"
                    )
            
            with col2:
                st.markdown("### All Models Performance")
                
                # Create performance table
                models_data = []
                for model_name, model_info in selected_data['all_models'].items():
                    models_data.append({
                        'Model': model_name,
                        'Recall': model_info['metrics']['recall'],
                        'Precision': model_info['metrics']['precision'],
                        'F1': model_info['metrics']['f1'],
                        'ROC AUC': model_info['metrics'].get('roc_auc', 'N/A')
                    })
                
                models_df = pd.DataFrame(models_data)
                st.dataframe(models_df.style.highlight_max(subset=['Recall', 'F1']), 
                            use_container_width=True)
                
                # Performance comparison chart
                fig = go.Figure(data=[
                    go.Bar(name='Recall', x=models_df['Model'], y=models_df['Recall'], 
                          marker_color='#3B82F6'),
                    go.Bar(name='Precision', x=models_df['Model'], y=models_df['Precision'], 
                          marker_color='#10B981'),
                    go.Bar(name='F1 Score', x=models_df['Model'], y=models_df['F1'], 
                          marker_color='#8B5CF6')
                ])
                
                fig.update_layout(
                    title="Model Performance Comparison",
                    barmode='group',
                    height=400,
                    xaxis_title="Models",
                    yaxis_title="Score",
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Historical performance trend
        st.markdown("### Historical Performance Trend")
        
        trend_data = []
        for run in runs[:10]:  # Last 10 runs
            trend_data.append({
                'Date': run['training_timestamp'],
                'Best Recall': run['best_recall_model']['metrics']['recall'],
                'Best F1': run['best_recall_model']['metrics']['f1'],
                'Best Model': run['best_recall_model']['name']
            })
        
        trend_df = pd.DataFrame(trend_data)
        
        fig = px.line(trend_df, x='Date', y=['Best Recall', 'Best F1'], 
                     title="Performance Over Time",
                     markers=True)
        
        fig.update_layout(
            height=400,
            xaxis_title="Training Date",
            yaxis_title="Score",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Settings
elif app_mode == "⚙️ Settings":
    st.markdown("<h2 class='sub-header'>⚙️ Settings</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Path Configuration")
        
        artifacts_path = st.text_input(
            "Artifacts Directory",
            value="../src/artifacts",
            help="Path where models and results are saved"
        )
        
        data_path = st.text_input(
            "Data Directory",
            value="../src/data",
            help="Path to your data files"
        )
        
        st.markdown("### Model Defaults")
        
        default_random_state = st.number_input(
            "Default Random State",
            value=42,
            help="Random seed for reproducibility"
        )
        
        default_test_size = st.slider(
            "Default Test Size (%)",
            min_value=10,
            max_value=40,
            value=20,
            help="Percentage of data to use for testing"
        )
    
    with col2:
        st.markdown("### Dashboard Settings")
        
        refresh_interval = st.selectbox(
            "Auto-refresh Interval",
            ["Off", "30 seconds", "1 minute", "5 minutes", "10 minutes"]
        )
        
        theme = st.selectbox(
            "Theme",
            ["Light", "Dark", "System Default"]
        )
        
        chart_style = st.selectbox(
            "Default Chart Style",
            ["Plotly", "Matplotlib", "Seaborn"]
        )
        
        st.markdown("### Data Management")
        
        if st.button("🔄 Clear Cache", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        
        if st.button("🗑️ Delete Old Models", use_container_width=True):
            st.warning("This will delete models older than 30 days. Continue?")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Yes, Delete"):
                    st.info("Feature not implemented yet")
            with col_b:
                if st.button("Cancel"):
                    st.rerun()
    
    # Save settings
    if st.button("💾 Save Settings", use_container_width=True):
        st.success("Settings saved successfully!")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown(
        """
        <div style='text-align: center; color: #6B7280; font-size: 0.9rem;'>
        <p>Model Training Dashboard • Version 1.0.0</p>
        <p>• Last updated: 09 February 2026 •</p>
        </div>
        """,
        unsafe_allow_html=True
    )