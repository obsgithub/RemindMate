import time
import requests
import re
import pandas as pd
import signal
import sys

# ======== Configuration ========
METRICS_URL = "http://YOUR_METRICS_ENDPOINT_HERE/metrics"  # e.g., "http://localhost:8000/metrics"
SAVE_PATH = "YOUR_OUTPUT_DATA.csv"
INTERVAL = 0.5

all_data = []

def parse_all_metrics(text):
    results = {'timestamp': time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}
    lines = text.split('\n')
    for line in lines:
        if line.startswith('#') or not line.strip():
            continue
        
        # Improved regex: Capture metric name, label pairs, and value
        # Format: name{label1="v1",label2="v2"} value
        match = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*?\})?\s+([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)$', line)
        if match:
            base_name = match.group(1)
            labels_str = match.group(2)
            value = float(match.group(3))
            
            # Process labels, making constructs like le="1.0" part of the column name
            full_name = base_name
            if labels_str:
                # Extract the 'le' label or other key labels to simplify column names
                le_match = re.search(r'le="([^"]+)"', labels_str)
                if le_match:
                    full_name = f"{base_name}_le_{le_match.group(1)}"
                else:
                    # If 'le' is not present, there might be other differentiating labels.
                    # We can choose to retain them to prevent name collisions,
                    # or simply record base_name_sum / base_name_count.
                    pass 
            
            results[full_name] = value
    return results

def handle_exit(sig, frame):
    print("\n[Stopping Collection] Processing and saving to disk...")
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(SAVE_PATH, index=False)
        print(f"Save complete: {SAVE_PATH}")
    sys.exit(0)

# Register the signal handler for Ctrl+C
signal.signal(signal.SIGINT, handle_exit)

def start_polling():
    print("Polling all metrics... Press Ctrl+C to stop and generate CSV")
    while True:
        try:
            resp = requests.get(METRICS_URL, timeout=0.5)
            if resp.status_code == 200:
                all_data.append(parse_all_metrics(resp.text))
        except:
            pass
        time.sleep(INTERVAL)

if __name__ == "__main__":
    # Example usage for demonstration purposes only.
    # Requires a valid metrics endpoint to execute properly.
    
    # start_polling()
    pass