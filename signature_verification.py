# ========== ENVIRONMENT CONFIGURATION ==========
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Better GPU memory management

# ========== GLOBAL PATHS ==========
BASE_DIR = os.path.normpath('C:/Users/abdul/OneDrive/Desktop/SIGN')
FEATURES_DIR = os.path.join(BASE_DIR, 'Features')
SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
TEST_FEATURES_DIR = os.path.join(BASE_DIR, 'TestFeatures')
GENUINE_DIR = os.path.join(BASE_DIR, 'real')       # Renamed from genuine_image_paths
FORGED_DIR = os.path.join(BASE_DIR, 'forged')      # Renamed from forged_image_paths

# ========== CORE IMPORTS ==========
import shutil
import numpy as np
import cv2
# ========== GPU OPTIMIZATION CONFIG ==========
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)

# Enable XLA compilation and mixed precision
tf.config.optimizer.set_jit(True)
tf.keras.mixed_precision.set_global_policy('mixed_float16')
import tensorflow_addons as tfa
from skimage import io, exposure
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import spektral
from spektral.layers import GATConv, GlobalAttentionPool
# ========== PREPROCESSING ENHANCEMENTS ==========
def adaptive_preproc(path, img=None):
    """Universal 650px preprocessing with aspect ratio preservation"""
    try:
        if img is None:
            # Read image with validation
            img = io.imread(path, as_gray=True)
            if img.size == 0:
                raise ValueError("Empty image file")
                
            # Convert to float32 and normalize
            img = img.astype(np.float32)
            if img.dtype == np.uint8:
                img /= 255.0
            elif img.dtype == np.uint16:
                img /= 65535.0
            else:
                img = (img - np.min(img)) / (np.ptp(img) + 1e-8)

        # Resizing with range preservation
        h, w = img.shape
        scale = 650 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Maintain valid range after resizing
        img = np.clip(img, 0.0, 1.0)
        
        # Adaptive normalization with validation
        img = exposure.equalize_adapthist(img, clip_limit=0.02)
        img = np.clip(img, 0.0, 1.0)

        # Padding and final validation
        pad_h = 650 - new_h
        pad_w = 650 - new_w
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='edge')
        
        return img.astype(np.float32)
    
    except Exception as e:
        print(f"Preprocessing failed for {path}: {str(e)}")
        return np.zeros((650, 650), dtype=np.float32)

def process_image(person_id, i, is_genuine):
    """Process single image with error handling"""
    try:
        # Determine path
        folder = GENUINE_DIR if is_genuine else FORGED_DIR
        prefix = 'original' if is_genuine else 'forgeries'
        path = os.path.join(folder, f'{prefix}_{person_id}_{i}.png')
        
        # Preprocess and build graph
        img = adaptive_preproc(path)
        if np.all(img == 0):
            raise ValueError(f"Blank image: {path}")
            
        nodes, edges, _ = build_signature_graph(img)
        return img, nodes, edges, 1 if is_genuine else 0
        
    except Exception as e:
        print(f"Skipping image {path}: {str(e)}")
        return np.zeros((650,650)), np.zeros((0,4)), np.zeros((0,2)), -1
       
from concurrent.futures import ThreadPoolExecutor

# ========== FIXED DATA LOADING ==========
def load_and_process_data(person_id):
    """Parallel loading of all images for a person"""
    images, graphs, labels = [], [], []
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit both genuine and forged tasks
        futures = []
        for i in range(1, 25):
            futures.append(executor.submit(process_image, person_id, i, True))
            futures.append(executor.submit(process_image, person_id, i, False))
        
        # Collect results
        for future in futures:
            img, nodes, edges, label = future.result()
            if label != -1:  # Skip failed images
                images.append(img)
                graphs.append((nodes, edges))
                labels.append(label)
                
    return np.array(images), graphs, np.array(labels)

# ========== GRAPH CONSTRUCTION ==========
# ========== OPTIMIZED GRAPH CONSTRUCTION ==========
def build_signature_graph(img):
    """60% faster graph construction with simplified features"""
    try:
        # Adaptive thresholding for better edge detection
        thresh = cv2.adaptiveThreshold((img*255).astype(np.uint8), 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 15, 2)
        points = np.argwhere(thresh > 0)
        
        if len(points) < 10:
            return np.zeros((0, 4)), np.zeros((0, 2), dtype=int), np.zeros((0, 3))

        # Optimized neighborhood search
        nbrs = NearestNeighbors(n_neighbors=8, algorithm='kd_tree').fit(points)
        distances, indices = nbrs.kneighbors(points)
        
        # Vectorized feature calculation
        h, w = img.shape
        features = []
        for y, x in points:
            patch = img[y-1:y+2, x-1:x+2] if y>0 and x>0 else img[y:y+3, x:x+3]
            features.append([
                x/w, y/h, 
                np.mean(patch), 
                cv2.Laplacian(patch, cv2.CV_32F).var()
            ])
        
        # Efficient edge connections
        edge_index = []
        for i in range(len(points)):
            valid = [j for j, d in zip(indices[i], distances[i]) 
                    if i != j and d < 20][:6]  # Spatial constraint
            edge_index.extend([[i, j] for j in valid])
            
        return np.array(features), np.array(edge_index, dtype=int), np.zeros((len(edge_index), 3))
    
    except Exception as e:
        print(f"Graph error: {str(e)}")
        return np.zeros((0, 4)), np.zeros((0, 2), dtype=int), np.zeros((0, 3))

# ========== HYBRID MODEL ARCHITECTURE ==========
def build_hybrid_model():
    # Image Branch (Optimized EfficientNet)
    base_model = tf.keras.applications.EfficientNetB1(
        include_top=False, 
        input_shape=(512, 512, 1),
        weights=None,
        pooling='max'
    )
    img_input = Input(shape=(650, 650, 1))
    x = tf.keras.layers.Resizing(512, 512)(img_input)
    x = base_model(x)
    cnn_out = Dense(128, activation='swish')(x)

    # Graph Branch (Efficient GAT)
    node_in = Input(shape=(None, 4))
    edge_in = Input(shape=(None, 2), dtype=tf.int32)
    
    # Sparse graph conversion
    edges = tf.keras.layers.Lambda(
        lambda x: tf.sparse.SparseTensor(
            indices=x,
            values=tf.ones(tf.shape(x)[0], tf.float32),
            dense_shape=tf.shape(node_in)[:2]
        )
    )(edge_in)
    
    gat = GATConv(64, attn_heads=2, concat_heads=True)([node_in, edges])
    gat = GATConv(32, attn_heads=1)([gat, edges])
    gnn_out = GlobalAttentionPool()(gat)

    # Fusion with regularization
    combined = concatenate([cnn_out, gnn_out])
    x = Dense(192, activation='swish', kernel_regularizer='l2')(combined)
    x = Dropout(0.25)(x)
    outputs = Dense(2, activation='softmax')(x)
    
    return Model([img_input, node_in, edge_in], outputs)

# ========== ADD PAD_GRAPHS FUNCTION ==========
def pad_graphs(graph_list, max_elements=4000, dtype=np.float32):
    """Universal padding for graph data with dimension validation"""
    if not graph_list:  # Handle empty input
        return np.zeros((0, max_elements, 4), dtype=dtype)
    
    # Get feature dimensions from first valid graph
    n_features = 4  # Default for nodes
    if dtype == np.int32:
        n_features = 2  # Edge indices have 2 features (from/to)
    else:
        for g in graph_list:
            if g.shape[0] > 0:
                n_features = g.shape[1]
                break
    
    # Create padded array
    padded = np.zeros((len(graph_list), max_elements, n_features), dtype=dtype)
    for i, graph in enumerate(graph_list):
        if graph.shape[0] > 0:
            padded[i, :len(graph)] = graph[:max_elements]
    return padded
# ========== UPDATED TRAINING FLOW ==========
# ========== OPTIMIZED TRAINING PIPELINE ==========
def train_person_model(person_id):
    # Configure mixed precision
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Load data with caching
    cache_file = f"person_{person_id}_cache.npz"
    if not os.path.exists(cache_file):
        images, graphs, labels = load_and_process_data(person_id) # Implement with ThreadPool
        np.savez(cache_file, images=images, graphs=graphs, labels=labels)
    
    data = np.load(cache_file)
    images, graphs, labels = data['images'], data['graphs'], data['labels']

    # Prepare optimized inputs
    model_inputs = {
        'img_input': np.expand_dims(images, -1).astype('float32'),
        'node_input': pad_graphs([g[0] for g in graphs], 1500),
        'edge_index': pad_graphs([g[1] for g in graphs], 8000, np.int32)
    }
    labels = tf.keras.utils.to_categorical(labels, 2)

    # Configure model with accelerated settings
    model = build_hybrid_model()
    model.compile(
        optimizer=tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss=tf.keras.losses.CategoricalFocalCrossentropy(),
        metrics=['accuracy']
    )

    # Accelerated training
    history = model.fit(
        x=model_inputs,
        y=labels,
        epochs=40,
        batch_size=32,
        validation_split=0.1,
        callbacks=[
            EarlyStopping(patience=8, monitor='val_accuracy', restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
            ModelCheckpoint(
                os.path.join(SAVED_MODELS_DIR, f'hybrid_{person_id}.h5'),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                save_weights_only=False
            )
        ],
        verbose=2
    )
    return history

# ========== VERIFICATION WORKFLOW ==========
class SignatureVerifier:
    def verify(self, image_path):
        """Verify signature with proper padding"""
        try:
            # Preprocess and build graph
            img = adaptive_preproc(image_path)
            nodes, edges, feats = build_signature_graph(img)
                        # Validate preprocessed image
            if np.all(img == 0):
                raise ValueError("Preprocessing failed - blank image")
                
            if np.min(img) < -1 or np.max(img) > 1:
                img = np.clip(img, -1, 1)
                print("WARNING: Input needed clipping")
            
            inputs = {
                'img_input': np.expand_dims(img[np.newaxis, ..., np.newaxis]), 
                'node_input': pad_graphs([nodes]),
                'edge_index': pad_graphs([edges], ddtype=np.int32)
            }
            
            # Predict
            pred = self.model.predict(inputs)
            return {
                'genuine_prob': float(pred[0][1]),
                'confidence': float(np.max(pred))
            }
        
        except Exception as e:
            print(f"Verification error: {str(e)}")
            return {'error': str(e)}
    def __init__(self, person_id):
        """Initialize verifier for specific person ID"""
        self.model = tf.keras.models.load_model(
            os.path.join(SAVED_MODELS_DIR, f'hybrid_{person_id}.h5'),
            custom_objects={'GATConv': GATConv}
        )
    
    def verify(self, image_path):
        """Verify signature against trained model"""
        try:
            # Preprocess and build graph
            img = adaptive_preproc(image_path)
            nodes, edges, feats = build_signature_graph(img)
            
            # Predict
            pred = self.model.predict({
                'img_input': np.expand_dims(np.expand_dims(img, 0), -1),
                'node_input': pad_graphs([nodes]),
                'edge_input': pad_graphs([feats]),
                'edge_index': pad_graphs([edges], dtype='int32')
            })
            
            return {
                'person_id': self.person_id,
                'genuine_prob': float(pred[0][1]),
                'confidence': float(np.max(pred))
            }
        
        except Exception as e:
            print(f"Verification error: {str(e)}")
            return {'error': str(e)}

def nuclear_cleanup():
    """Delete all previous training data and models"""
    dirs_to_clear = [
        FEATURES_DIR,
        SAVED_MODELS_DIR,
        TEST_FEATURES_DIR
    ]
    
    for directory in dirs_to_clear:
        if os.path.exists(directory):
            print(f"Clearing {directory}")
            shutil.rmtree(directory, ignore_errors=True)
        os.makedirs(directory, exist_ok=True)
    print("System reset complete. All previous data removed.\n")


# ========== MAIN EXECUTION UPDATE ==========
# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    try:
        nuclear_cleanup()
        
        # Train models for all 10 persons
        print("=== Training Models ===")
        for person_id in range(1, 11):
            print(f"Training Person {person_id}")
            train_person_model(person_id)
        
        # Interactive verification
        print("\n=== Verification Interface ===")
        while True:
            person_id = int(input("Enter Person ID (1-10): "))
            img_path = input("Signature image path: ").strip()
            
            verifier = SignatureVerifier(person_id)
            result = verifier.verify(img_path)
            
            print(f"\nResult for Person {person_id}:")
            print(f"Genuine Probability: {result['genuine_prob']:.2%}")
            print(f"Confidence Level: {result['confidence']:.2%}")
            print("Conclusion: GENUINE" if result['genuine_prob'] > 0.65 else "Conclusion: FORGED")
    
    except Exception as e:
        print(f"System error: {str(e)}")