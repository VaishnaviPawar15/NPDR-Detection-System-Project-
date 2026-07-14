import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def crop_image(img, tol=7):
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    mask = gray > tol
    return img[np.ix_(mask.any(1), mask.any(0))]

def gaussian_filter(img):
    
    return cv2.GaussianBlur(img, (5, 5), sigmaX=1)

def illumination_correction(img):
    
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=10)
    corrected = cv2.addWeighted(img, 4, blur, -4, 128)
    return corrected

INPUT_FOLDER = r"D:\Python Programs\NPDR\Severe"  
OUTPUT_FOLDER = r"D:\Python Programs\NPDR\Preprocessed_Severe_DR" 

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

IMG_SIZE = 224

print("\n Starting Gaussian preprocessing...\n")

for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
        img_path = os.path.join(INPUT_FOLDER, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Could not read {filename}, skipping...")
            continue

        cropped = crop_image(img)

        resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))

        gaussian_img = gaussian_filter(resized)

        illum_corrected = illumination_correction(gaussian_img)

        normalized = np.clip(illum_corrected / 255.0, 0, 1)
        normalized_uint8 = (normalized * 255).astype(np.uint8)

        save_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(save_path, cv2.cvtColor(normalized_uint8, cv2.COLOR_RGB2BGR))

        print(f" Processed and saved: {filename}")

print("\n All images have been preprocessed using Gaussian filter!\n")

sample_file = os.listdir(INPUT_FOLDER)[0]
original = cv2.imread(os.path.join(INPUT_FOLDER, sample_file))
processed = cv2.imread(os.path.join(OUTPUT_FOLDER, sample_file))

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title("Original Sample")

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
plt.title("Gaussian Preprocessed Sample")

plt.show()
