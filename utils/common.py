"""
Common Utilities Module

This module provides common utilities used throughout the application.
It centralizes shared functionality to reduce code duplication.
"""

import os
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import pandas as pd
import numpy as np

# Set up logging
def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging with consistent formatting.
    
    Args:
        name: Logger name
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    return logger

# Create a default logger for this module
logger = setup_logging(__name__)

def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
    """
    os.makedirs(directory_path, exist_ok=True)
    logger.debug(f"Ensured directory exists: {directory_path}")

def load_json_file(file_path: str, default: Any = None) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        default: Default value to return if file doesn't exist or is invalid
    
    Returns:
        Loaded data or default value
    """
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return default
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {str(e)}")
        return default

def save_json_file(file_path: str, data: Any, indent: int = 2) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        file_path: Path to the JSON file
        data: Data to save
        indent: JSON indentation level
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(file_path)
        ensure_directory_exists(directory)
        
        # Save file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent, default=json_serializer)
        
        logger.debug(f"Saved JSON file: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {str(e)}")
        return False

def json_serializer(obj: Any) -> Any:
    """
    JSON serializer for objects not serializable by default json code.
    
    Args:
        obj: Object to serialize
    
    Returns:
        Serialized object
    """
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if pd.isna(obj):
        return None
    raise TypeError(f"Type {type(obj)} not serializable")

def safe_divide(numerator: Union[int, float], denominator: Union[int, float], default: Union[int, float] = 0) -> Union[int, float]:
    """
    Safely divide two numbers, returning a default value if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value to return if denominator is zero
    
    Returns:
        Result of division or default value
    """
    return numerator / denominator if denominator != 0 else default

def retry_operation(operation: Callable, max_retries: int = 3, retry_delay: int = 1, 
                   exceptions: Tuple = (Exception,), logger: Optional[logging.Logger] = None) -> Any:
    """
    Retry an operation multiple times before giving up.
    
    Args:
        operation: Function to retry
        max_retries: Maximum number of retries
        retry_delay: Delay between retries in seconds
        exceptions: Exceptions to catch and retry
        logger: Logger to use (uses module logger if None)
    
    Returns:
        Result of the operation
    
    Raises:
        The last exception encountered if all retries fail
    """
    import time
    
    log = logger or globals()['logger']
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return operation()
        except exceptions as e:
            last_exception = e
            log.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    log.error(f"All {max_retries} attempts failed")
    raise last_exception

def get_nested_dict_value(d: Dict, keys: List[str], default: Any = None) -> Any:
    """
    Safely get a nested value from a dictionary.
    
    Args:
        d: Dictionary to get value from
        keys: List of keys to traverse
        default: Default value to return if key doesn't exist
    
    Returns:
        Value or default
    """
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
