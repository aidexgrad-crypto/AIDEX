"""
Test script to verify data cleaning functionality
"""
import pandas as pd
import numpy as np
import requests
import json
import os

# Backend URL
BASE_URL = "http://localhost:8000"

def create_test_dataset():
    """Create a test dataset with various data quality issues"""
    print("\n" + "="*80)
    print("CREATING TEST DATASET WITH QUALITY ISSUES")
    print("="*80)
    
    # Create dataset with issues
    data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'age': [25, 30, np.nan, 45, 50, 25, 30, 35, 40, 45, 50, np.nan],  # Missing values
        'income': [50000, 60000, 70000, 80000, 90000, 50000, 60000, 70000, 80000, 90000, 100000, 110000],
        'constant_col': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Constant column
        'mostly_missing': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 'value', np.nan, np.nan, np.nan, np.nan],  # >40% missing
        'category': ['A', 'B', np.nan, 'A', 'B', 'A', 'B', 'A', 'B', 'A', np.nan, 'B'],  # Missing categorical
        'target': [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1]
    }
    
    df = pd.DataFrame(data)
    
    # Add duplicate rows
    df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)
    
    print(f"\n📊 Test Dataset Created:")
    print(f"   - Shape: {df.shape}")
    print(f"   - Columns: {list(df.columns)}")
    print(f"   - Missing values in 'age': {df['age'].isnull().sum()}")
    print(f"   - Missing values in 'mostly_missing': {df['mostly_missing'].isnull().sum()} ({df['mostly_missing'].isnull().sum()/len(df)*100:.1f}%)")
    print(f"   - Duplicates: {df.duplicated().sum()}")
    print(f"   - Constant column 'constant_col': unique values = {df['constant_col'].nunique()}")
    
    return df

def test_cleaning_endpoint(df):
    """Test the /data/clean endpoint"""
    print("\n" + "="*80)
    print("TESTING /data/clean ENDPOINT")
    print("="*80)
    
    # Prepare request
    payload = {
        "data": df.replace({np.nan: None}).to_dict('records'),
        "target_column": "target",
        "protected_columns": ["id", "target"]
    }
    
    print(f"\n📤 Sending cleaning request...")
    print(f"   - Rows: {len(df)}")
    print(f"   - Columns: {len(df.columns)}")
    print(f"   - Target column: 'target'")
    print(f"   - Protected columns: ['id', 'target']")
    
    try:
        response = requests.post(f"{BASE_URL}/data/clean", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get("status") == "success":
                print(f"\n✅ CLEANING SUCCESSFUL!")
                
                summary = result.get("summary", {})
                print(f"\n📋 Cleaning Summary:")
                print(f"   - Before: {summary.get('before_rows')} rows, {summary.get('before_cols')} columns")
                print(f"   - After: {summary.get('after_rows')} rows, {summary.get('after_cols')} columns")
                print(f"   - Duplicates removed: {summary.get('removed_duplicates')}")
                print(f"   - Missing values filled: {summary.get('filled_missing')}")
                
                if summary.get('dropped_columns'):
                    print(f"\n🗑️  Dropped Columns:")
                    for col_info in summary['dropped_columns']:
                        print(f"   - {col_info['name']}: {col_info['reason']}")
                
                if summary.get('notes'):
                    print(f"\n📝 Notes:")
                    for note in summary['notes']:
                        print(f"   - {note}")
                
                print(f"\n💾 Cleaned file saved to: {result.get('cleaned_file_path')}")
                print(f"   - Total rows in file: {result.get('total_rows')}")
                print(f"   - Preview rows returned: {result.get('preview_rows')}")
                
                # Verify the cleaned data
                cleaned_data = pd.DataFrame(result['cleaned_data'])
                print(f"\n🔍 Verification:")
                print(f"   - Cleaned data shape: {cleaned_data.shape}")
                print(f"   - Missing values: {cleaned_data.isnull().sum().sum()}")
                print(f"   - Columns present: {list(cleaned_data.columns)}")
                print(f"   - 'constant_col' removed: {'constant_col' not in cleaned_data.columns}")
                print(f"   - 'mostly_missing' removed: {'mostly_missing' not in cleaned_data.columns}")
                
                # Check if target column is preserved
                if 'target' in cleaned_data.columns:
                    print(f"   - Target column preserved: ✅")
                    print(f"   - Target unique values: {cleaned_data['target'].unique()}")
                else:
                    print(f"   - Target column preserved: ❌ ERROR!")
                
                return True, result
            else:
                print(f"\n❌ CLEANING FAILED!")
                print(f"   Error: {result.get('error')}")
                return False, result
        else:
            print(f"\n❌ HTTP ERROR!")
            print(f"   Status code: {response.status_code}")
            print(f"   Response: {response.text}")
            return False, None
            
    except Exception as e:
        print(f"\n❌ EXCEPTION OCCURRED!")
        print(f"   Error: {str(e)}")
        return False, None

def test_quality_engine_analyze():
    """Test the DataQualityEngine analyze endpoint"""
    print("\n" + "="*80)
    print("TESTING /data-quality/analyze ENDPOINT")
    print("="*80)
    
    # Create test CSV file
    df = create_test_dataset()
    test_file = "test_dataset.csv"
    df.to_csv(test_file, index=False)
    
    print(f"\n📤 Uploading file for analysis...")
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': (test_file, f, 'text/csv')}
            response = requests.post(f"{BASE_URL}/data-quality/analyze", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ ANALYSIS SUCCESSFUL!")
            
            print(f"\n📊 Dataset Info:")
            print(f"   - Shape: {result.get('shape')}")
            print(f"   - Columns: {len(result.get('columns', []))}")
            
            if result.get('missing_values'):
                print(f"\n🔍 Missing Values:")
                for col, info in result['missing_values'].items():
                    print(f"   - {col}: {info['count']} ({info['percentage']:.1f}%)")
            
            print(f"\n📋 Duplicates: {result.get('duplicates')}")
            
            return True, result
        else:
            print(f"\n❌ HTTP ERROR!")
            print(f"   Status code: {response.status_code}")
            return False, None
    
    except Exception as e:
        print(f"\n❌ EXCEPTION OCCURRED!")
        print(f"   Error: {str(e)}")
        return False, None
    
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)

def main():
    """Run all cleaning tests"""
    print("\n" + "="*80)
    print("AIDEX DATA CLEANING VERIFICATION TEST")
    print("="*80)
    
    # Check if backend is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        print(f"\n✅ Backend is running at {BASE_URL}")
    except:
        print(f"\n❌ ERROR: Backend is not running at {BASE_URL}")
        print(f"   Please make sure the backend is started first.")
        return
    
    # Test 1: Clean endpoint
    df = create_test_dataset()
    success1, result1 = test_cleaning_endpoint(df)
    
    # Test 2: Quality engine analyze
    success2, result2 = test_quality_engine_analyze()
    
    # Final summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"   /data/clean endpoint: {'✅ PASSED' if success1 else '❌ FAILED'}")
    print(f"   /data-quality/analyze endpoint: {'✅ PASSED' if success2 else '❌ FAILED'}")
    
    if success1 and success2:
        print(f"\n🎉 ALL TESTS PASSED! Data cleaning is working properly.")
    else:
        print(f"\n⚠️  SOME TESTS FAILED! Please check the errors above.")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
