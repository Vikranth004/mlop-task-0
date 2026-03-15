import argparse
import logging
import sys
import yaml
import numpy as np
import os
import pandas as pd
import io
import time
import json

def setup_logging(log_file_path):
    """Configures logging to output to both a file and the console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def load_and_validate_config(config_path, logger):
    """Loads YAML and validates required fields."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        # The PDF requires these specific keys 
        required_keys = ['seed', 'window', 'version']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
                
        logger.info(f"Config loaded and validated: seed={config['seed']}, window={config['window']}, version={config['version']}")
        return config
        
    except FileNotFoundError:
        logger.error(f"Config file not found at path: {config_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Config validation failed: {e}")
        sys.exit(1)

def load_and_validate_data(data_path, logger):
    """Loads the CSV and validates its structure according to requirements."""
    logger.info(f"Attempting to load data from {data_path}")
    
    # 1. Check if file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Input file not found: {data_path}")
        
    # 2. Check if file is completely empty (0 bytes)
    if os.path.getsize(data_path) == 0:
        raise ValueError(f"Input file is empty: {data_path}")
        
    try:
        # 3. MLOps TRAP FIX: The dataset is wrapped in double quotes.
        # We clean the raw text in memory before pandas tries to parse it.
        with open(data_path, 'r', encoding='utf-8') as f:
            clean_csv_text = f.read().replace('"', '')
            
        # Load the cleaned text into pandas
        df = pd.read_csv(io.StringIO(clean_csv_text))
        
        # Standardize column names (lowercase, remove spaces)
        df.columns = df.columns.str.strip().str.lower()
        
    except Exception as e:
        raise ValueError(f"Invalid CSV format: {e}")
        
    # 4. Check if the DataFrame has actual rows of data
    if df.empty:
        raise ValueError("CSV file loaded, but it contains no data rows.")
        
    # 5. Validate the required 'close' column exists
    if 'close' not in df.columns:
        raise ValueError(f"Missing required column: 'close'. Found columns: {list(df.columns)}")
        
    logger.info(f"Data loaded successfully. Rows validated: {len(df)}")
    return df

def process_data(df, window, logger):
    """Calculates rolling mean and generates the binary trading signal."""
    logger.info(f"Starting data processing. Calculating rolling mean with window size {window}.")
    
    # 1. Calculate the rolling mean on the 'close' column
    # The PDF states: "define how you handle the first window-1 rows"
    # We will use min_periods=1 to calculate a mean even for the first few rows, 
    # instead of filling them with NaNs, which is standard practice in these pipelines.
    df['rolling_mean'] = df['close'].rolling(window=window, min_periods=1).mean()
    
    # 2. Generate the binary signal
    # signal = 1 if close > rolling_mean, else 0
    df['signal'] = np.where(df['close'] > df['rolling_mean'], 1, 0)
    
    logger.info("Signal generation complete.")
    return df

def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="MLOps Batch Job")
    parser.add_argument("--input", required=True, help="Path to input CSV data")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--output", required=True, help="Path to output metrics JSON")
    parser.add_argument("--log-file", required=True, dest="log_file", help="Path to run.log")
    
    args = parser.parse_args()
    
    logger = setup_logging(args.log_file)
    logger.info("Job start timestamp")
    
    # Initialize default error metrics
    metrics = {
        "version": "unknown",
        "status": "error",
        "error_message": "Initialization failed"
    }
    
    try:
        config = load_and_validate_config(args.config, logger)
        metrics["version"] = config["version"] # Update version once config is loaded
        
        np.random.seed(config['seed'])
        logger.info("Deterministic seed set successfully.")

        # Load and process data
        df = load_and_validate_data(args.input, logger)
        df = process_data(df, config['window'], logger)
        
        # Calculate final metrics
        rows_processed = len(df)
        signal_rate = float(df['signal'].mean()) 
        latency_ms = int((time.time() - start_time) * 1000)
        
        logger.info(f"Metrics Summary - Rows: {rows_processed}, Signal Rate: {signal_rate:.4f}, Latency: {latency_ms}ms")
        
        # Update metrics dictionary for success
        metrics.update({
            "status": "success",
            "rows_processed": rows_processed,
            "metric": "signal_rate",
            "value": round(signal_rate, 4),
            "latency_ms": latency_ms,
            "seed": config['seed']
        })
        metrics.pop("error_message", None) # Remove error message on success
        
        logger.info("Job completed successfully.")
        
    except Exception as e:
        logger.error(f"Job failed: {e}")
        metrics["error_message"] = str(e)
        # We don't use sys.exit(1) right here because we want the 'finally' block to run
        
    finally:
        # This block ALWAYS runs, ensuring the JSON is written even if it fails
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics written to {args.output}")
        
        print(json.dumps(metrics, indent=4))
        # Exit with correct code based on status
        if metrics["status"] == "error":
            sys.exit(1)

if __name__ == "__main__":
    main()