import pandas as pd
from dash import Dash, dcc, html
import plotly.graph_objs as go
import os
import sys

# ================= Configuration =================
# Mode selection: 'web' (Real-time web dashboard) or 'html' (Save as offline file)
OUTPUT_MODE = 'html' 

# Placeholders for data input and report output
CSV_FILE = "YOUR_METRICS_DATA.csv"
SAVE_PATH = "YOUR_REPORT_OUTPUT.html"
# =================================================

def generate_vllm_report():
    if not os.path.exists(CSV_FILE):
        print(f"Error: File not found - {CSV_FILE}")
        sys.exit()

    # Data preprocessing
    df = pd.read_csv(CSV_FILE)
    all_cols = [c for c in df.columns if c != 'timestamp']
    groups = {}
    for c in all_cols:
        base = c.split('_le_')[0]
        if base not in groups: 
            groups[base] = []
        groups[base].append(c)

    # 1. Core plotting logic: Generate all figure objects
    figures = []
    for base_name, cols in sorted(groups.items()):
        is_histogram = any('_le_' in c for c in cols)
        fig = go.Figure()
        
        if is_histogram:
            def get_le_val(name):
                val = name.split('_le_')[-1]
                return float('inf') if val == '+Inf' else float(val)
            
            sorted_cols = sorted(cols, key=get_le_val)
            for i, col in enumerate(sorted_cols):
                display_vals = df[col] if i == 0 else df[col] - df[sorted_cols[i-1]]
                fig.add_trace(go.Scatter(
                    x=df['timestamp'], y=display_vals,
                    name=f"Bucket: {col.split('_le_')[-1]}",
                    mode='lines', stackgroup='one',
                    line=dict(width=0.5), hoverinfo='x+y+name'
                ))
            fig.update_layout(title=f"Distribution Trend (Stacked Area): {base_name}")
        else:
            for col in cols:
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df[col], name=col, mode='lines'))
            fig.update_layout(title=f"Metric: {base_name}")

        fig.update_layout(
            xaxis_title="Time", yaxis_title="Value",
            height=450, hovermode="x unified",
            template="plotly_white", margin=dict(l=40, r=40, t=60, b=40)
        )
        figures.append(fig)

    # 2. Execute output based on the selected mode
    if OUTPUT_MODE == 'web':
        # --- Real-time Web Mode ---
        print("Starting in [Real-time Web] mode...")
        app = Dash(__name__)
        
        layout_children = [html.H1("vLLM Comprehensive Metrics Real-time Dashboard", 
                                  style={'textAlign': 'center', 'fontFamily': 'sans-serif', 'padding': '20px'})]
        
        for fig in figures:
            layout_children.append(html.Div([dcc.Graph(figure=fig)], 
                                            style={'margin': '20px', 'padding': '10px', 
                                                   'backgroundColor': 'white', 'borderRadius': '8px',
                                                   'boxShadow': '0px 4px 12px rgba(0,0,0,0.1)'}))
        
        app.layout = html.Div(layout_children, style={'backgroundColor': '#f9f9f9', 'minHeight': '100vh'})
        # Note: Port is hardcoded. Modify as needed.
        app.run(host='0.0.0.0', port=8051)

    elif OUTPUT_MODE == 'html':
        # --- Offline HTML Mode ---
        print(f"Exporting in [Offline HTML] mode to: {SAVE_PATH}...")
        
        html_content = [
            f'<html><head><meta charset="utf-8"/><title>vLLM Metrics Report</title></head>',
            f'<body style="background-color: #f9f9f9; font-family: sans-serif; padding: 20px;">',
            f'<h1 style="text-align: center; color: #333;">vLLM Comprehensive Metrics Offline Analysis Report</h1>'
        ]
        
        for fig in figures:
            # Export figure as a div snippet
            chart_div = fig.to_html(full_html=False, include_plotlyjs='cdn', 
                                    config={'responsive': True})
            html_content.append(f'<div style="margin-bottom: 30px; background: white; padding: 15px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">{chart_div}</div>')
        
        html_content.append('</body></html>')
        
        with open(SAVE_PATH, 'w', encoding='utf-8') as f:
            f.write("\n".join(html_content))
        
        print("Export successful! You can open the file directly in a browser to view it.")

    else:
        print(f"Error: Unknown OUTPUT_MODE '{OUTPUT_MODE}'. Please set to 'web' or 'html'.")


if __name__ == "__main__":
    # Example usage for demonstration purposes only.
    # Requires a valid CSV metrics file to execute properly.
    
    # generate_vllm_report()
    pass