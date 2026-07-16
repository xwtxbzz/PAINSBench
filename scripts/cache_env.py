import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR

env_file = os.path.join(PROCESSED_DIR, "env_labels_puda.npy")
if os.path.exists(env_file):
    el = np.load(env_file)
    print(f"[OK] Env labels: {len(el)} samples, dist={np.bincount(el).tolist()}")
else:
    print("Computing env labels...")
    dd = np.load(os.path.join(PROCESSED_DIR, "features_full.npz"), allow_pickle=True)
    fp = dd["X"][:,:256]
    from sklearn.decomposition import PCA
    pca = PCA(n_components=5, random_state=42)
    proj = pca.fit_transform(fp)
    el = np.argmax(proj, axis=1)
    np.save(env_file, el)
    print(f"  Done: dist={np.bincount(el).tolist()}")
print("All caches ready.")
