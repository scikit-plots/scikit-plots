# Force garbage collection
import gc; gc.collect()
import os
import shutil

def clear_keras_cache():
    """
    Clears the cached Keras model weights stored in the ~/.keras/models/ directory.
    
    Be cautious: This will delete all cached models, which may require re-downloading them later.
    """
    cache_dir = os.path.expanduser('~/.keras/models/')
    
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)  # Delete the entire folder with cached models
            print(f"Successfully cleared Keras cache at: {cache_dir}")
        except Exception as e:
            print(f"Error while clearing the Keras cache: {e}")
    else:
        print(f"No cache directory found at: {cache_dir}")

if __name__ == "__main__":
    # Call the function to clear cache when this module is run directly
    clear_keras_cache()