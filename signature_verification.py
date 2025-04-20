
# ========== IMPORTS CLEANED ==========
import numpy as np
import os
import shutil 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from skimage.measure import regionprops, label
from skimage import io
from skimage.filters import threshold_otsu, threshold_local
from skimage import exposure
from traceback import format_exc
import pandas as pd
from itertools import chain
from time import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras import layers, utils
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
import joblib
 

# ========== GLOBAL PATHS ==========
BASE_DIR = os.path.normpath('C:/Users/abdul/OneDrive/Desktop/SIGN')
FEATURES_DIR = os.path.join(BASE_DIR, 'Features')
SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
TEST_FEATURES_DIR = os.path.join(BASE_DIR, 'TestFeatures')
genuine_image_paths = os.path.join(BASE_DIR, 'real')
forged_image_paths = os.path.join(BASE_DIR, 'forged')


def rgbgrey(img):
    # Converts rgb to grayscale
    greyimg = np.zeros((img.shape[0], img.shape[1]))
    for row in range(len(img)):
        for col in range(len(img[row])):
            greyimg[row][col] = np.average(img[row][col])
    return greyimg

def greybin(img):
    # Converts grayscale to binary
    blur_radius = 0.8
    img = ndimage.gaussian_filter(img, blur_radius)  # to remove small components or noise
#     img = ndimage.binary_erosion(img).astype(img.dtype)
    thres = threshold_otsu(img)
    binimg = img > thres
    binimg = np.logical_not(binimg)
    return binimg

def preproc(path, img=None, display=True):
    try:
        if img is None:
            img = io.imread(path, as_gray=True)  # Force grayscale reading
            if img.size == 0:
                return np.zeros((100, 100))

        # Adaptive contrast enhancement
        img = exposure.equalize_adapthist(img, clip_limit=0.03)

        # Adaptive thresholding
        thresh = threshold_local(img, block_size=25, method='gaussian')
        binary = img > thresh
        binary = np.logical_not(binary)  # Invert for dark signatures

        # Improved contour detection
        labeled = label(binary)
        regions = regionprops(labeled)
        
        if regions:
            # Dynamic area threshold (5% of image area)
            min_area = 0.05 * binary.size
            valid_regions = [r for r in regions if r.area >= min_area]
            
            if valid_regions:
                largest = max(valid_regions, key=lambda x: x.area)
                minr, minc, maxr, maxc = largest.bbox
                cropped = binary[minr:maxr, minc:maxc]
                
                # Pad while maintaining aspect ratio
                return resize_with_padding(cropped, (100, 100))

        return resize_with_padding(binary, (100, 100))

    except Exception as e:
        if display:
            print(f"Preprocessing error: {str(e)}")
        return np.zeros((100, 100))
    
def resize_with_padding(img, target_size=(100, 100)):
    # Maintain aspect ratio with zero padding
    ratio = min(target_size[0]/img.shape[0], target_size[1]/img.shape[1])
    new_size = [int(round(img.shape[0]*ratio)), int(round(img.shape[1]*ratio))]
    resized = ndimage.zoom(img, ratio, order=0)
    
    pad_y = target_size[0] - resized.shape[0]
    pad_x = target_size[1] - resized.shape[1]
    
    return np.pad(resized, ((0, pad_y), (0, pad_x)), mode='constant')

def Ratio(img):
    """Compatability wrapper for legacy code"""
    return np.mean(img) if isinstance(img, np.ndarray) else 0.0

def Centroid(img):
    numOfWhites = 0
    a = np.array([0,0])
    for row in range(len(img)):
        for col in range(len(img[0])):
            if img[row][col]==True:
                b = np.array([row,col])
                a = np.add(a,b)
                numOfWhites += 1
    rowcols = np.array([img.shape[0], img.shape[1]])
    centroid = a/numOfWhites
    centroid = centroid/rowcols
    return centroid[0], centroid[1]

# ================== IMPROVEMENT 1: Better Region Detection ==================

def EccentricitySolidity(img):
    labeled_img = label(img.astype("int8"))
    regions = regionprops(labeled_img)
    if not regions:  # Handle case with no regions
        return 0.0, 0.0
    largest_region = max(regions, key=lambda x: x.area)
    return largest_region.eccentricity, largest_region.solidity


def SkewKurtosis(img):
    h,w = img.shape
    x = range(w)  # cols value
    y = range(h)  # rows value
    #calculate projections along the x and y axes
    xp = np.sum(img,axis=0)
    yp = np.sum(img,axis=1)
    #centroid
    cx = np.sum(x*xp)/np.sum(xp)
    cy = np.sum(y*yp)/np.sum(yp)
    #standard deviation
    x2 = (x-cx)**2
    y2 = (y-cy)**2
    sx = np.sqrt(np.sum(x2*xp)/np.sum(img))
    sy = np.sqrt(np.sum(y2*yp)/np.sum(img))
    
    #skewness
    x3 = (x-cx)**3
    y3 = (y-cy)**3
    skewx = np.sum(xp*x3)/(np.sum(img) * sx**3)
    skewy = np.sum(yp*y3)/(np.sum(img) * sy**3)

    #Kurtosis
    x4 = (x-cx)**4
    y4 = (y-cy)**4
    # 3 is subtracted to calculate relative to the normal distribution
    kurtx = np.sum(xp*x4)/(np.sum(img) * sx**4) - 3
    kurty = np.sum(yp*y4)/(np.sum(img) * sy**4) - 3

    return (skewx , skewy), (kurtx, kurty)


def getFeatures(path, img=None, display=False):
    if img is None:
        img = mpimg.imread(path)
    img = preproc(path, display=display)
    ratio = Ratio(img)
    centroid = Centroid(img)
    eccentricity, solidity = EccentricitySolidity(img)
    skewness, kurtosis = SkewKurtosis(img)
    retVal = (ratio, centroid, eccentricity, solidity, skewness, kurtosis)
    return retVal

def getCSVFeatures(path, img=None, display=False):
    try:
        img = preproc(path, display=display)
        binary_img = img > 0.5
        
        # Feature calculations with enhanced validation
        ratio = np.mean(binary_img)
        height, width = binary_img.shape
        
        # Centroid calculation with fallback
        white_pixels = np.argwhere(binary_img)
        if len(white_pixels) == 0:
            cent_y, cent_x = 0.5, 0.5
        else:
            cent_y, cent_x = np.mean(white_pixels, axis=0)
            cent_y /= height
            cent_x /= width

        # Region properties with safety checks
        labeled = label(binary_img.astype("uint8"))
        regions = regionprops(labeled)
        eccentricity = solidity = 0.0
        if regions:
            main_region = max(regions, key=lambda x: x.area, default=None)
            if main_region and main_region.area >= 50:
                eccentricity = main_region.eccentricity
                solidity = main_region.solidity

        # Skew/kurtosis with numerical stability
        skew_x = skew_y = 0.0
        if np.var(binary_img) > 1e-5:  # Avoid division by zero
            try:
                (skew_x, skew_y), _ = SkewKurtosis(binary_img)
            except:
                pass

        # GLCM features with fallback
        contrast = homogeneity = 0.0
        try:
            from skimage.feature import graycomatrix, graycoprops
            glcm = graycomatrix((img*255).astype('uint8'), 
                              distances=[5], angles=[0], symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        except Exception as e:
            if display:
                print(f"Texture features failed: {str(e)}")
        # Add final validation before return
        if all(v == 0 for v in (contrast, homogeneity)) and (cent_y, cent_x) == (0.5, 0.5):
            raise ValueError("Invalid feature extraction - likely no signature detected")
        
        return (round(ratio,4), round(cent_y,4), round(cent_x,4),
                round(eccentricity,4), round(solidity,4),
                round(contrast,4), round(homogeneity,4),
                round(skew_x,4), round(skew_y,4))
    
    except Exception as e:
        print(f"Feature error: {str(e)}")
        return (0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  

def verify_image_shapes():
    for person in range(1, 11):
        # Check genuine signatures
        for i in chain(range(1, 25), range(1, 25)):  # Update ranges as needed
            path = os.path.join(genuine_image_paths, f'original_{person}_{i}.png')
            if os.path.exists(path):
                try:
                    img = mpimg.imread(path)
                    if len(img.shape) not in [2, 3]:
                        print(f"Invalid shape {img.shape} in {path}")
                except Exception as e:
                    print(f"Corrupted file {path}: {str(e)}")
                    
        # Check forged signatures
        for i in range(1, 25):
            path = os.path.join(forged_image_paths, f'forgeries_{person}_{i}.png')
            if os.path.exists(path):
                try:
                    img = mpimg.imread(path)
                    if len(img.shape) not in [2, 3]:
                        print(f"Invalid shape {img.shape} in {path}")
                except Exception as e:
                    print(f"Corrupted file {path}: {str(e)}")


def makeCSV():
    # Add this verification
    if not os.path.exists(genuine_image_paths):
        raise FileNotFoundError(f"Genuine images path missing: {genuine_image_paths}")
    if not os.path.exists(forged_image_paths):
        raise FileNotFoundError(f"Forged images path missing: {forged_image_paths}")
    
    # Verify preprocessing works before CSV creation
    test_path = os.path.join(genuine_image_paths, "original_1_1.png")
    test_output = preproc(test_path, display=False)
    # Check if we get at least 5% white pixels
    if np.mean(test_output) < 0.05:
        print("Validation Warning:")
        print(f"- White pixels ratio: {np.mean(test_output):.2%}")
        print("- Trying to proceed with fallback...")

    # Create directories using global paths
    train_dir = os.path.join(FEATURES_DIR, 'Training')
    test_dir = os.path.join(FEATURES_DIR, 'Testing')
    
    os.makedirs(train_dir, exist_ok=True)  # Creates directory if it doesn't exist
    os.makedirs(test_dir, exist_ok=True)
    print("Directories verified/created.")

    # Genuine and forged paths
    gpath = genuine_image_paths
    fpath = forged_image_paths

    for person in range(1, 11):
        train_file = os.path.join(train_dir, f'training_{person}.csv')
        test_file = os.path.join(test_dir, f'testing_{person}.csv')

        # Skip if both files already exist
        if os.path.exists(train_file) and os.path.exists(test_file):
            print(f'Skipping Person {person}: CSVs already exist.')
            continue

        # Generate Training CSV only if missing
        if not os.path.exists(train_file):
            print(f'Creating training CSV for Person {person}...')
            with open(train_file, 'w') as handle:
                handle.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y,output\n')
                # Add genuine samples (1-15)
                for i in range(1, 16):
                    source = os.path.join(gpath, f'original_{person}_{i}.png')
                    features = getCSVFeatures(path=source)
                    handle.write(','.join(map(str, features)) + ',1\n')
                # Add forged samples (1-15)
                for i in range(1, 16):
                    source = os.path.join(fpath, f'forgeries_{person}_{i}.png')
                    features = getCSVFeatures(path=source)
                    handle.write(','.join(map(str, features)) + ',0\n')

        # Generate Testing CSV only if missing
        if not os.path.exists(test_file):
            print(f'Creating testing CSV for Person {person}...')
            with open(test_file, 'w') as handle:
                handle.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y,output\n')
                # Add genuine samples (16-24)
                for i in range(16, 25):
                    source = os.path.join(gpath, f'original_{person}_{i}.png')
                    features = getCSVFeatures(path=source)
                    handle.write(','.join(map(str, features)) + ',1\n')
                # Add forged samples (16-24)
                for i in range(16, 25):
                    source = os.path.join(fpath, f'forgeries_{person}_{i}.png')
                    features = getCSVFeatures(path=source)
                    handle.write(','.join(map(str, features)) + ',0\n')



def testing(test_image_path, person_id):
    """Process a test signature image with robust validation and proper scaling"""
    test_csv = None  # Initialize for cleanup
    try:
        # ========== Initial Validation ==========
        if not os.path.exists(test_image_path):
            raise FileNotFoundError(f"Image not found: {test_image_path}")
            
        if not 1 <= person_id <= 10:
            raise ValueError("Person ID must be between 1-10")

        # ========== Feature Extraction ==========
        features = getCSVFeatures(test_image_path, display=False)
        
        # Validate feature extraction results
        if features is None:
            raise ValueError("Feature extraction failed completely")
            
        if len(features) != 9:
            raise ValueError(f"Invalid feature count: {len(features)} (expected 9)")
            
        # Convert to numpy array with proper shape
        features_array = np.array(features).reshape(1, -1)
        
        # Check for default values indicating failure
        if (features_array[0, 1:3] == [0.5, 0.5]).all() and features_array[0, 0] < 0.05:
            raise ValueError("Signature detection failed - centroid at image center")

        # ========== Scaler Handling ==========
        scaler_path = os.path.join(SAVED_MODELS_DIR, f'scaler_{person_id}.pkl')
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found for person {person_id}")
            
        scaler = joblib.load(scaler_path)
        
        # Validate scaler compatibility
        if scaler.n_features_in_ != features_array.shape[1]:
            raise ValueError(f"Scaler expects {scaler.n_features_in_} features, got {features_array.shape[1]}")

        # ========== Feature Scaling ==========
        scaled_features = scaler.transform(features_array)
        
        # Post-scaling validation
        if np.any(np.isnan(scaled_features)) or np.any(np.isinf(scaled_features)):
            raise ValueError("Invalid values after scaling")

        # ========== CSV Handling ==========
        test_dir = os.path.join(BASE_DIR, 'TestFeatures')
        os.makedirs(test_dir, exist_ok=True)
        test_csv = os.path.join(test_dir, f'test_{person_id}.csv')

        # Write RAW features with headers
        pd.DataFrame(features_array, 
                    columns=['ratio','cent_y','cent_x','eccentricity',
                             'solidity','contrast','homogeneity',
                             'skew_x','skew_y']).to_csv(test_csv, index=False)

        # Final validation
        if not pd.read_csv(test_csv).shape[0] == 1:
            raise IOError("CSV creation failed - no valid data written")

        return test_csv

    except Exception as e:
        print(f"Testing failed: {str(e)}")
        if test_csv and os.path.exists(test_csv):
            try:
                os.remove(test_csv)
            except Exception as cleanup_err:
                print(f"Cleanup failed: {str(cleanup_err)}")
        return None

n_input = 9

# ================== IMPROVEMENT 2: Feature Normalization ==================


def readCSV(train_path, test_path, type2=False, scaler=None):
    """Enhanced CSV reader with robust validation and error handling"""
    
    def validate_data(data, source):
        """Helper function for data validation"""
        if data.size == 0:
            raise ValueError(f"Empty data array from {source}")
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            raise ValueError(f"Invalid values detected in {source}")
        return data

    try:
        # ========== Train Data Processing ==========
        # Read and validate training features
        df_train = pd.read_csv(train_path, usecols=range(n_input))
        if df_train.empty:
            raise ValueError(f"Empty training CSV: {train_path}")
            
        train_input = validate_data(
            df_train.values.astype(np.float32, copy=False),
            "training features"
        )

        # Scaler handling with fallback
        if scaler is None:
            scaler = StandardScaler()
            train_input = scaler.fit_transform(train_input)
        else:  # Verification mode
            train_input = scaler.transform(train_input)

        # Read and validate training labels
        df_train_labels = pd.read_csv(train_path, usecols=(n_input,))
        if df_train_labels.empty:
            raise ValueError(f"Empty training labels in {train_path}")
            
        corr_train = tf.keras.utils.to_categorical(
            df_train_labels.values.flatten().astype(int), 
            2
        )

        # ========== Test Data Processing ==========
        # Read and validate test features
        df_test = pd.read_csv(test_path, usecols=range(n_input))
        if df_test.empty:
            raise ValueError(f"Empty test CSV: {test_path}")
            
        test_input = validate_data(
            df_test.values.astype(np.float32, copy=False),
            "test features"
        )
        
        # Ensure 2D array for scaler
        if test_input.ndim == 1:
            test_input = test_input.reshape(-1, n_input)
            
        test_input = scaler.transform(test_input)

        # ========== Return Handling ==========
        if not type2:
            # Training mode: process test labels
            df_test_labels = pd.read_csv(test_path, usecols=(n_input,))
            if df_test_labels.empty:
                raise ValueError(f"Empty test labels in {test_path}")
                
            corr_test = tf.keras.utils.to_categorical(
                df_test_labels.values.flatten().astype(int),
                2
            )
            return train_input, corr_train, test_input, corr_test, scaler
            
        return train_input, corr_train, test_input, scaler

    except Exception as e:
        print(f"CSV Read Error: {str(e)}")
        print(f"Train path: {train_path}")
        print(f"Test path: {test_path}")
        print(f"Type2 mode: {type2}")
        raise

# Parameters
learning_rate = 0.0001        # Reduced learning rate
training_epochs = 1000        # Increased epochs         # Reduced from 1000 (early stopping will handle termination)
display_step = 1

n_hidden_1 = 128
n_hidden_2 = 64
n_hidden_3 = 32
n_classes = 2 # no. of classes (genuine or forged)

def multilayer_perceptron(x):
    # Layer 1 with full variable scoping
    with tf.variable_scope('layer1'):
        w1 = tf.get_variable('weights', initializer=tf.keras.initializers.he_normal()([n_input, n_hidden_1]))
        b1 = tf.get_variable('biases', initializer=tf.zeros([n_hidden_1]))
        layer = tf.nn.relu(tf.matmul(x, w1) + b1)
        layer = tf.keras.layers.BatchNormalization(name='bn1')(layer)
        layer = tf.nn.dropout(layer, rate=0.3)

    # Layer 2 with full variable scoping    
    with tf.variable_scope('layer2'):
        w2 = tf.get_variable('weights', initializer=tf.keras.initializers.he_normal()([n_hidden_1, n_hidden_2]))
        b2 = tf.get_variable('biases', initializer=tf.zeros([n_hidden_2]))
        layer = tf.nn.relu(tf.matmul(layer, w2) + b2)
        layer = tf.keras.layers.BatchNormalization(name='bn2')(layer)
        layer = tf.nn.dropout(layer, rate=0.3)

    # Layer 3 with full variable scoping
    with tf.variable_scope('layer3'):
        w3 = tf.get_variable('weights', initializer=tf.keras.initializers.he_normal()([n_hidden_2, n_hidden_3]))
        b3 = tf.get_variable('biases', initializer=tf.zeros([n_hidden_3]))
        layer = tf.nn.relu(tf.matmul(layer, w3) + b3)

    # Output layer
    with tf.variable_scope('output'):
        w_out = tf.get_variable('weights', initializer=tf.keras.initializers.glorot_uniform()([n_hidden_3, n_classes]))
        b_out = tf.get_variable('biases', initializer=tf.zeros([n_classes]))
        out_layer = tf.matmul(layer, w_out) + b_out

    return out_layer

def evaluate(train_path, test_path, type2=False):
    # Reset graph and create new session for each evaluation
    tf.reset_default_graph()
    
    # Get person ID early
    person_id = os.path.basename(train_path).split('_')[1].split('.')[0]
    model_dir = SAVED_MODELS_DIR
    scaler_path = os.path.join(model_dir, f'scaler_{person_id}.pkl')

    # Load data with scaler handling
    if not type2:
        # Training mode: fit and save scaler
        train_input, corr_train, test_input, corr_test, scaler = readCSV(train_path, test_path)
        joblib.dump(scaler, scaler_path)  # Save scaler for future use
    else:
        # Verification mode: load existing scaler
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found for person {person_id}. Train first!")            
        scaler = joblib.load(scaler_path)
        # Load test data using stored scaler
        train_input, corr_train, test_input, _ = readCSV(train_path, test_path, type2=True, scaler=scaler)
        
        if test_input.ndim == 1:
            test_input = test_input.reshape(1, -1)

    # Model persistence setup
    model_dir = SAVED_MODELS_DIR
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'model_{person_id}.ckpt')

    # Define model architecture within the new graph context
    with tf.Graph().as_default():
        X = tf.placeholder(tf.float32, [None, n_input])
        Y = tf.placeholder(tf.float32, [None, n_classes])
        keep_prob = tf.placeholder(tf.float32)

        # Define network parameters within this graph context
        weights = {
            'h1': tf.Variable(tf.keras.initializers.he_normal()([n_input, n_hidden_1])),
            'h2': tf.Variable(tf.keras.initializers.he_normal()([n_hidden_1, n_hidden_2])),
            'h3': tf.Variable(tf.keras.initializers.he_normal()([n_hidden_2, n_hidden_3])),
            'out': tf.Variable(tf.keras.initializers.glorot_uniform()([n_hidden_3, n_classes]))
        }

        biases = {
            'b1': tf.Variable(tf.zeros([n_hidden_1])),
            'b2': tf.Variable(tf.zeros([n_hidden_2])),
            'b3': tf.Variable(tf.zeros([n_hidden_3])),
            'out': tf.Variable(tf.zeros([n_classes]))
        }

        

        # Build model components
        logits = multilayer_perceptron(X)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y)
        loss_op = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)
        pred = tf.nn.softmax(logits)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            saver = tf.train.Saver()
            
            if os.path.exists(model_path + '.index'):
                saver.restore(sess, model_path)
                print(f"Loaded trained model for person {person_id}")
            else:
                sess.run(init)
                best_val_acc = 0.0
                patience = 25
                no_improve = 0
                
                try:
                    for epoch in range(training_epochs):
                        # Mini-batch training
                        for i in range(0, len(train_input), 64):
                            batch_x = train_input[i:i+64]
                            batch_y = corr_train[i:i+64]
                            sess.run(train_op, feed_dict={
                                X: batch_x, 
                                Y: batch_y, 
                                keep_prob: 0.7
                            })

                        # Validation
                        val_acc = accuracy.eval({
                            X: test_input,
                            Y: corr_test,
                            keep_prob: 1.0
                        })

                        # Early stopping
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            no_improve = 0
                            saver.save(sess, model_path)
                        else:
                            no_improve += 1

                        if no_improve >= patience:
                            print(f"Early stopping at epoch {epoch+1}")
                            break
                finally:
                    saver.restore(sess, model_path)
                    print(f"Training complete for person {person_id}")

            # Prediction/Evaluation
            confidence_threshold = 0.7
            
            if not type2:
                train_acc = accuracy.eval({
                    X: train_input, 
                    Y: corr_train, 
                    keep_prob: 1.0
                })
                test_acc = accuracy.eval({
                    X: test_input, 
                    Y: corr_test, 
                    keep_prob: 1.0
                })
                return train_acc, test_acc
            else:
                prediction = pred.eval({
                    X: test_input, 
                    keep_prob: 1.0
                })
                genuine_prob = prediction[0][1]
                
                if genuine_prob > confidence_threshold:
                    print(f'Genuine Signature ({genuine_prob:.2%} confidence)')
                    return True
                elif genuine_prob < (1 - confidence_threshold):
                    print(f'Forged Signature ({1 - genuine_prob:.2%} confidence)')
                    return False
                else:
                    print(f'Uncertain Prediction ({genuine_prob:.2%})')
                    return None

def trainAndTest(rate=0.001, epochs=500, neurons=128, display=True):
    start = time()

    # ========== UPDATED PARAMETERS ==========
    global learning_rate, training_epochs, n_hidden_1, n_hidden_2, n_hidden_3
    
    # Training parameters
    learning_rate = rate       # Reduced learning rate
    training_epochs = epochs   # Number of epochs
    
    # Network architecture (Updated to 32-32-16)
    n_hidden_1 = neurons      # First hidden layer (32 neurons)
    n_hidden_2 = 64           # Second hidden layer (32 neurons)
    n_hidden_3 = 32           # Third hidden layer (16 neurons)

    train_avg, test_avg = 0, 0
    n = 10  # Number of persons to process

    for i in range(1, n+1):
        if display:
            print(f"Processing Person {i}...")  # Remove zero-padding from print
            
        # Use numeric person ID without zero-padding
        person_id = str(i)
        # Fix hardcoded paths
        modified_train_path = os.path.join(FEATURES_DIR, 'Training', f'training_{person_id}.csv')
        modified_test_path = os.path.join(FEATURES_DIR, 'Testing', f'testing_{person_id}.csv')
        
        train_score, test_score = evaluate(modified_train_path, modified_test_path)
        train_avg += train_score
        test_avg += test_score

    if display:
        print("\n=== Final Results ===")
        print(f"Network Architecture: {n_hidden_1}-{n_hidden_2}-{n_hidden_3}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Average Training Accuracy: {train_avg/n:.4f}")
        print(f"Average Testing Accuracy: {test_avg/n:.4f}")
        print(f"Average Time per Person: {(time()-start)/n:.2f}s")
    
    return train_avg/n, test_avg/n, (time()-start)/n

def nuclear_cleanup():
    dirs = [SAVED_MODELS_DIR, FEATURES_DIR, TEST_FEATURES_DIR]
    for d in dirs:
        if os.path.exists(d):
            print(f"Removing {d}")
            shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
    print("All previous data erased. Starting fresh!\n")

# Run at program start


# ========== MAIN EXECUTION FLOW ==========
# ========== MAIN EXECUTION FLOW ==========
if __name__ == "__main__":
    try:
         # Step 0: Clean all previous data
        print("=== Initializing System ===")
        nuclear_cleanup() # Add this line
        
        
        # ==============================
        # Step 1: Verify all images
        print("\n=== Verifying Image Integrity ===")
        verify_image_shapes()
        
        # Step 2: Generate feature CSVs
        print("\n=== Generating Training Data ===")
        makeCSV()
        
        # Model training
        print("\n=== Training Models ===")
        start_time = time()
        train_acc, test_acc, _ = trainAndTest(
            rate=0.001, 
            epochs=1000,
            neurons=128,
            display=True
        )
        print(f"\nFinal Metrics | Train: {train_acc:.2%} | Test: {test_acc:.2%}")
        print(f"Total Training Time: {time()-start_time:.2f}s")

        # Step 4: Verification Interface
        print("\n=== Signature Verification System ===")
        while True:
            try:
                # Get person ID
                person_id = int(input("\nEnter person's ID (1-10) or 0 to exit: "))
                if person_id == 0:
                    print("Exiting system...")
                    break
                if not 1 <= person_id <= 10:
                    print("Please enter between 1-10")
                    continue
                
                # Get signature image
                test_image = input("Enter signature image path: ").strip()
                if not os.path.exists(test_image):
                    print("File not found! Please check path.")
                    continue
                
                # Process test image with error handling
                test_csv = testing(test_image, person_id)
                
                if test_csv is None:
                    print("Failed to process signature. Possible reasons:")
                    print("- Image format not supported")
                    print("- No signature detected in image")
                    print("- Low quality/unreadable signature")
                    continue
                
                # Run verification
                train_csv = os.path.join(FEATURES_DIR, 'Training', f'training_{person_id}.csv')
                model_file = os.path.join(SAVED_MODELS_DIR, f'model_{person_id}.ckpt.index')

                # Fix 1: Corrected training data check
                if not os.path.exists(train_csv):
                    print(f"Training data missing for person {person_id}")
                    continue

                # Fix 2: Proper model existence check and training
                if not os.path.exists(model_file):
                    print(f"Model not found. Training first...")
                    print(f"Training model for person {person_id}...")
                    # Train only this specific person's model
                    modified_train_path = os.path.join(FEATURES_DIR, 'Training', f'training_{person_id}.csv')
                    modified_test_path = os.path.join(FEATURES_DIR, 'Testing', f'testing_{person_id}.csv')
                    evaluate(modified_train_path, modified_test_path)  # Single-person training

                print("\n=== Verifying Signature ===")
                try:
                    result = evaluate(train_csv, test_csv, type2=True)
                    if result is None:
                        print("Verification inconclusive - low confidence")
                    else:
                        status = "Genuine" if result else "Forged"
                        print(f"\nVerification Result: {status} signature")
                except Exception as e:
                    print(f"Verification failed: {str(e)}")
                    print("Common causes:")
                    print("- Corrupted test data file")
                    print("- Model/scaler version mismatch")
                    print("- Invalid feature scaling")
            
            except ValueError:
                print("Invalid input. Numbers only please.")
            except Exception as e:
                print(f"System error: {str(e)}")
    
    except KeyboardInterrupt:
        print("\nSystem shutdown requested. Exiting...")
    finally:
        print("\n=== System Cleanup ===")
        # Cleanup temporary test files
        try:
            shutil.rmtree(TEST_FEATURES_DIR, ignore_errors=True)
            print("Temporary files cleaned")
        except Exception as e:
            print(f"Cleanup error: {str(e)}")