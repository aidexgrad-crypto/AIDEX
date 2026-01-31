"""Simple test for /data/clean endpoint"""
import requests
import pandas as pd
import numpy as np

# Create test data
data = {
    'id': [1, 2, 3, 4, 5],
    'age': [25, 30, np.nan, 45, 50],
    'income': [50000, 60000, 70000, 80000, 90000],
    'constant_col': [1, 1, 1, 1, 1],
    'target': [0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

print("Testing /data/clean endpoint...")
print(f"Input shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")

# Send request - convert NaN to None for JSON compatibility
payload = {
    "data": df.replace({np.nan: None}).to_dict('records'),
    "target_column": "target",
    "protected_columns": ["id", "target"]
}

try:
    response = requests.post("http://localhost:8000/data/clean", json=payload, timeout=30)
    
    print(f"\nResponse status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ SUCCESS!")
        print(f"Status: {result.get('status')}")
        if result.get('status') == 'success':
            summary = result.get('summary', {})
            print(f"\nSummary:")
            print(f"  Before: {summary.get('before_rows')} rows, {summary.get('before_cols')} cols")
            print(f"  After: {summary.get('after_rows')} rows, {summary.get('after_cols')} cols")
            print(f"  Duplicates removed: {summary.get('removed_duplicates')}")
            print(f"  Missing filled: {summary.get('filled_missing')}")
            print(f"  Dropped columns: {summary.get('dropped_columns')}")
        else:
            print(f"Error: {result.get('error')}")
    else:
        print(f"\n❌ HTTP Error: {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"\n❌ Exception: {e}")
    import traceback
    traceback.print_exc()
