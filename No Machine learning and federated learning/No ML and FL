import multiprocessing
from tqdm import tqdm
import numpy as np
import glob
import cv2
import sys
import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import argparse

class ImagePreprocessor(object):
    def __init__(self, root_dir: str, save_dir: str, img_size: int, tolerance: int = 10, remove_outer_pixels: float = 0.0):
        """
        Preprocess images for kaggle competitions and general training purposes.
        """
        if remove_outer_pixels > 0.50:
            print("ERROR: eroding more than 50% of image")
            raise InterruptedError

        self.root_dir = root_dir
        self.img_size = img_size
        self.tolerance = tolerance
        self.remove_outer_pixels = remove_outer_pixels

        self.images = glob.glob(f"{self.root_dir}/*.png") + glob.glob(
            f"{self.root_dir}/*.jpeg") + glob.glob(f"{self.root_dir}/*.jpg")
        self.save_dir = os.path.join(root_dir, save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.total_count = len(self.images)

    # different preprocessing methods
    @staticmethod
    def light_sensitivity_reducer(img: np.ndarray, alpha: int = 4, beta: int = -4, gamma: int = 128):
        """smooth image and apply ben's preprocessing"""
        return cv2.addWeighted(img, alpha, cv2.GaussianBlur(img, (0, 0), 10), beta, gamma)

    @staticmethod
    def outer_pixels_remover(img: np.ndarray, scale: float):
        """remove outer/boundary pixels of image"""
        scale_2 = scale / 2.0
        miny = int(img.shape[0]*scale_2)
        maxy = int(img.shape[0]-miny)
        minx = int(img.shape[1]*scale_2)
        maxx = int(img.shape[1]-minx)
        return img[miny:maxy, minx:maxx]

    @staticmethod
    def scale_image(img: np.ndarray, img_size: int):
        """resize image based on given scale"""
        return cv2.resize(img, (img_size, img_size))

    @staticmethod
    def pitch_black_remover(img: np.ndarray, tolerance: int = 10):
        """remove black pixels in image edges"""
        if img.ndim == 2:
            img_mask = img > tolerance
            return img[np.ix_(img_mask.any(1), img_mask.any(0))]
        elif img.ndim == 3:
            greyed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_mask = greyed > tolerance
            img_1 = img[:, :, 0][np.ix_(img_mask.any(1), img_mask.any(0))]
            if img_1.shape[0] == 0:
                return img
            img_2 = img[:, :, 1][np.ix_(img_mask.any(1), img_mask.any(0))]
            img_3 = img[:, :, 2][np.ix_(img_mask.any(1), img_mask.any(0))]
            return np.stack([img_1, img_2, img_3], axis=-1)
        else:
            print("Image has more than 3 dimensions")
            raise InterruptedError

    def preprocess_image(self, img):
        """Preprocess single image"""
        img = self.pitch_black_remover(img, tolerance=self.tolerance)
        img = self.scale_image(img, img_size=self.img_size)
        img = self.light_sensitivity_reducer(img)
        if self.remove_outer_pixels > 0.0:
            img = self.outer_pixels_remover(img, self.remove_outer_pixels)
        return img

class RuleBasedClassifier:
    """Simple rule-based classifier - NO machine learning!"""
    
    def __init__(self):
        print("Initializing rule-based classifier...")
        
    def extract_features(self, img):
        """Extract features from image (without machine learning)"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 1. Red region (hemorrhages) detection
        # In HSV, red is in 0-10 and 170-180 range
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask_red1 + mask_red2
        
        # 2. Yellow region (exudates) detection
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # 3. White region (hard exudates) detection
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Calculate features
        features = {
            'red_area': np.sum(red_mask > 0) / (img.shape[0] * img.shape[1]),
            'yellow_area': np.sum(yellow_mask > 0) / (img.shape[0] * img.shape[1]),
            'white_area': np.sum(white_mask > 0) / (img.shape[0] * img.shape[1]),
            'avg_brightness': np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
            'contrast': np.std(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        }
        
        return features
    
    def classify_by_rules(self, features):
        """Classify based on rules (levels 0-4)"""
        red_area = features['red_area']
        yellow_area = features['yellow_area']
        white_area = features['white_area']
        brightness = features['avg_brightness']
        
        # Simple rule-based judgment
        if red_area < 0.001 and yellow_area < 0.001 and white_area < 0.001:
            return 0  # No DR
        
        elif red_area < 0.005 and yellow_area < 0.003 and white_area < 0.002:
            return 1  # Mild DR
        
        elif red_area < 0.01 or yellow_area < 0.01 or white_area < 0.01:
            return 2  # Moderate DR
        
        elif red_area < 0.05 or yellow_area < 0.05 or white_area < 0.05:
            return 3  # Severe DR
        
        else:
            return 4  # Proliferative DR

def evaluate_classifier():
    """Evaluate rule-based classifier on ALL images"""
    print("=" * 60)
    print("APTOS Diabetic Retinopathy - Rule-Based Classifier Evaluation")
    print("=" * 60)
    
    # Data paths
    dataset_path = "Aptos2019"
    train_csv_path = os.path.join(dataset_path, "train.csv")
    train_images_path = os.path.join(dataset_path, "train_images")
    
    # Load label data
    train_df = pd.read_csv(train_csv_path)
    total_images = len(train_df)
    print(f"Total samples: {total_images}")
    
    # Use ALL images for testing (not just 100)
    test_df = train_df  # Now using all images
    print(f"Testing samples: {len(test_df)}")
    
    # Initialize classifier and preprocessor
    classifier = RuleBasedClassifier()
    preprocessor = ImagePreprocessor(
        root_dir=train_images_path,
        save_dir="processed",
        img_size=224,
        tolerance=10,
        remove_outer_pixels=0.0
    )
    
    true_labels = []
    predicted_labels = []
    processed_count = 0
    
    print("\nStarting image processing and classification...")
    
    # Use tqdm for progress bar
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing images"):
        try:
            # Load image
            img_path = os.path.join(train_images_path, row['id_code'] + '.png')
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Cannot load image: {img_path}")
                continue
            
            # Preprocess
            processed_img = preprocessor.preprocess_image(img)
            
            # Extract features
            features = classifier.extract_features(processed_img)
            
            # Classify based on rules
            pred_label = classifier.classify_by_rules(features)
            
            # Record results
            true_labels.append(row['diagnosis'])
            predicted_labels.append(pred_label)
            processed_count += 1
            
            # Show progress every 100 images
            if processed_count % 100 == 0:
                print(f"Processed {processed_count}/{total_images} images")
                
        except Exception as e:
            print(f"Error processing image {row['id_code']}: {e}")
            continue
    
    # Calculate accuracy
    if len(true_labels) > 0:
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        print(f"Processed images: {len(true_labels)}")
        print(f"Successfully processed: {len(true_labels)}/{total_images} images")
        print(f"Rule-based classification accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        print("\nDetailed Classification Report:")
        print(classification_report(true_labels, predicted_labels, 
                                  target_names=['0-No DR', '1-Mild DR', '2-Moderate DR', 
                                                '3-Severe DR', '4-Proliferative DR']))
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        print("Confusion Matrix:")
        print(cm)
        
        return accuracy
    else:
        print("No images processed successfully!")
        return 0.0

def main():
    """Main function"""
    import warnings
    warnings.filterwarnings("ignore")
    
    accuracy = evaluate_classifier()
    
    print("\n" + "=" * 60)
    print("Benchmark Results Summary")
    print("=" * 60)
    print(f"Rule-based classification accuracy: {accuracy*100:.2f}%")
    print(f"Method: Traditional image processing + Rule-based judgment")
    print(f"Compliant with 'no ML' requirement: No machine learning algorithms used")
    print("=" * 60)

if __name__ == "__main__":
    main()
    
