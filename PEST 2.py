import os
import cv2
import random
import albumentations as A

# === CONFIGURATION ===
input_folder = "e:/PESTICIDE PROJECT/PESTICIDE AFFECTED"   # Use a folder of healthy or mixed images
output_folder = "Pesticide_Affected 2"
num_images = 2000  # Number of synthetic images to generate
image_size = (128, 128)  # Resize for model compatibility

# Create output directory
os.makedirs(output_folder, exist_ok=True)

# Define augmentation pipeline for pesticide damage
augment = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
    A.GaussNoise(var_limit=(20, 50), p=0.5),
    A.OneOf([
        A.Solarize(threshold=128, p=0.3),
        A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=50, val_shift_limit=30, p=0.5)
    ], p=0.7),
    A.Sharpen(alpha=(0.1, 0.4), lightness=(0.5, 1.0), p=0.4),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=10, p=0.2),
    A.RandomShadow(p=0.3),
    A.MotionBlur(p=0.2)
])

# Collect image files
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
random.shuffle(image_files)

# Limit to desired count
image_files = image_files[:num_images]

# Process and augment
for idx, file in enumerate(image_files):
    img_path = os.path.join(input_folder, file)
    image = cv2.imread(img_path)
    if image is None:
        continue

    image = cv2.resize(image, image_size)
    augmented = augment(image=image)['image']

    out_path = os.path.join(output_folder, f"pesticide_{idx+1}.jpg")
    cv2.imwrite(out_path, augmented)

    if idx % 100 == 0:
        print(f"[{idx}/{num_images}] Processed")

print("✅ Synthetic pesticide-affected images created.")
