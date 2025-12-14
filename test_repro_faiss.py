
import os
import shutil
from src.shared.infrastructure.ai.vector_db import VectorDB
import numpy as np

def test_vectordb_save_behavior():
    # Setup
    test_dir = "test_vectordb_artifacts"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir, exist_ok=True)
    db_path = os.path.join(test_dir, "test_db.faiss")
    
    print(f"Testing VectorDB save to: {db_path}")
    
    # Initialize DB
    db = VectorDB(dimension=4)
    
    # Add dummy data
    embeddings = [np.random.rand(4).astype('float32')]
    scenes = [{"item": 1, "start_time": 0, "end_time": 1}]
    db.add_scenes("vid1", scenes, embeddings)
    
    # Save
    db.save(db_path)
    
    # Check what exists
    exists_main = os.path.exists(db_path)
    exists_meta = os.path.exists(db_path + ".metadata")
    
    print(f"Exists Main '{db_path}': {exists_main}")
    print(f"Exists Meta '{db_path}.metadata': {exists_meta}")
    
    if exists_main and exists_meta:
        print("SUCCESS: VectorDB.save creates both main file and metadata sidecar.")
    else:
        print("FAILURE: Files missing.")
        
    # Verify Load
    print("Testing Load...")
    loaded_db = VectorDB.load(db_path, dimension=4)
    if loaded_db.is_video_indexed("vid1"):
         print("SUCCESS: Loaded DB contains vid1.")
    else:
         print("FAILURE: Loaded DB is empty or corrupt.")

    # Cleanup
    shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_vectordb_save_behavior()
