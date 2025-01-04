import streamlit as st
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from pathlib import Path  
# Functions for image transformations
def add_colored_patches(image, num_patches, patch_size_range):
    for _ in range(num_patches):
        x = np.random.randint(0, image.shape[1])
        y = np.random.randint(0, image.shape[0])
        w = np.random.randint(patch_size_range[0], patch_size_range[1])
        h = np.random.randint(patch_size_range[0], patch_size_range[1])
        color = [np.random.randint(0, 256) for _ in range(3)]  # Random RGB color
        alpha = np.random.uniform(0.3, 0.7)  # Transparency level
        overlay = image.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return image

def add_fading(image, intensity):
    fade_mask = np.zeros_like(image, dtype=np.float32)
    for channel in range(3):  # Iterate over color channels
        fade_mask[:, :, channel] = np.linspace(1, 1 - intensity, image.shape[1])
    faded_image = cv2.multiply(image.astype(np.float32) / 255.0, fade_mask)
    return (faded_image * 255).astype(np.uint8)

def add_stains(image, num_stains, stain_size_range):
    for _ in range(num_stains):
        center = (np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0]))
        radius = np.random.randint(stain_size_range[0], stain_size_range[1])
        color = [np.random.randint(0, 256) for _ in range(3)]  # Random RGB color
        alpha = np.random.uniform(0.3, 0.7)  # Transparency level
        overlay = image.copy()
        cv2.circle(overlay, center, radius, color, -1)
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return image


def add_gaussian_noise(image, noise_level):
    # Generate Gaussian noise
    noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
    # Add noise to the image
    noisy_image = image.astype(np.float32) + noise
    # Clip the values to ensure valid pixel range
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    total_pixels = image.size
    num_salt = int(salt_prob * total_pixels)
    num_pepper = int(pepper_prob * total_pixels)
    
    # Add salt noise
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy_image[salt_coords[0], salt_coords[1], :] = 255

    # Add pepper noise
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy_image[pepper_coords[0], pepper_coords[1], :] = 0
    
    return noisy_image

def add_blur(image, blur_level):
    if blur_level % 2 == 0:
        blur_level += 1  # Ensure the blur level is odd
    return cv2.GaussianBlur(image, (blur_level, blur_level), 0)

def add_scratches(image, num_scratches):
    for _ in range(num_scratches):
        start_point = (np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0]))
        end_point = (np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0]))
        # color = [np.random.randint(50, 255) for _ in range(3)]  # White scratches
        color = (255,255,255)
        thickness = np.random.randint(1, 2)
        # Scratch texture
        scratch_texture = np.random.normal(0, 50, image.shape[:2]).astype(np.uint8)
        scratch_texture = cv2.GaussianBlur(scratch_texture, (5, 5), 0)
        overlay = image.copy()
        cv2.line(overlay, start_point, end_point, color, thickness)
        image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
        
    return image

def adjust_brightness(image, brightness):
    return cv2.convertScaleAbs(image, alpha=1, beta=brightness)

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

# Functions for effects
def apply_motion_blur(image, kernel_size=15):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    return cv2.filter2D(image, -1, kernel)

def apply_edge_distortion(image, operation='erosion', kernel_size=5, iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    if operation == 'erosion':
        return cv2.erode(image, kernel, iterations=iterations)
    elif operation == 'dilation':
        return cv2.dilate(image, kernel, iterations=iterations)

def apply_vignetting(image, intensity=0.5):
    rows, cols = image.shape[:2]
    x = cv2.getGaussianKernel(cols, cols / 2)
    y = cv2.getGaussianKernel(rows, rows / 2)
    mask = y * x.T
    mask = mask / mask.max()
    vignette_image = np.copy(image)
    for i in range(3):
        vignette_image[:, :, i] = vignette_image[:, :, i] * (1 - intensity + intensity * mask)
    return vignette_image.astype(np.uint8)


# Streamlit app
st.title("Corrupto 🛠️: Advanced Dataset Damage Tool")
st.write("Easily add noise ⚙️ and effects to create a damaged dataset for computer vision tasks.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_name = Path(uploaded_file.name).stem
    original_extension = Path(uploaded_file.name).suffix
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, channels="BGR", caption="Original Image")

    # Sidebar for customization
    st.sidebar.header("Damage Effects")
    
    # Noise type
    noise_type = st.sidebar.selectbox("Choose Noise Type", ["None", "Gaussian", "Salt and Pepper"])
    
    # Apply transformations to the damaged_image in sequence
# Apply transformations to the damaged_image in sequence
    damaged_image = image.copy()

    # Gaussian Noise
    if noise_type == "Gaussian":
        noise_level = st.sidebar.slider("Gaussian Noise Level", 0.0, 100.0, 25.0)
        damaged_image = add_gaussian_noise(damaged_image, noise_level)

    # Salt and Pepper Noise
    if noise_type == "Salt and Pepper":
        salt_prob = st.sidebar.slider("Salt Probability", 0.0, 0.1, 0.01)
        pepper_prob = st.sidebar.slider("Pepper Probability", 0.0, 0.1, 0.01)
        damaged_image = add_salt_and_pepper_noise(damaged_image, salt_prob, pepper_prob)

    # Blur
    blur_level = st.sidebar.slider("Blur Level (Odd Number)", 1, 15, 5)
    damaged_image = add_blur(damaged_image, blur_level)

    # Colored Patches
    if st.sidebar.checkbox("Add Colored Patches"):
        num_patches = st.sidebar.slider("Number of Patches", 1, 50, 10)
        patch_size_range = st.sidebar.slider("Patch Size Range", 10, 100, (20, 50))
        damaged_image = add_colored_patches(damaged_image, num_patches, patch_size_range)

    # Fading
    if st.sidebar.checkbox("Add Fading"):
        intensity = st.sidebar.slider("Fading Intensity", 0.0, 1.0, 0.3)
        damaged_image = add_fading(damaged_image, intensity)

    # Stains
    if st.sidebar.checkbox("Add Stains"):
        num_stains = st.sidebar.slider("Number of Stains", 1, 20, 5)
        stain_size_range = st.sidebar.slider("Stain Size Range", 10, 100, (20, 50))
        damaged_image = add_stains(damaged_image, num_stains, stain_size_range)

    
    # Scratches
    if st.sidebar.checkbox("Add Scratches"):
        num_scratches = st.sidebar.slider("Number of Scratches", 1, 70, 10)
        damaged_image = add_scratches(damaged_image, num_scratches)

    # Brightness and Rotation
    brightness = st.sidebar.slider("Adjust Brightness", -100, 100, 0)
    damaged_image = adjust_brightness(damaged_image, brightness)
    rotation_angle = st.sidebar.slider("Rotate Image", -180, 180, 0)
    damaged_image = rotate_image(damaged_image, rotation_angle)

    # Sidebar for effects
    st.sidebar.header("Effects")

    # Motion Blur
    if st.sidebar.checkbox("Apply Motion Blur"):
        blur_kernel = st.sidebar.slider("Kernel Size (Odd Number)", 3, 31, 15, step=2)
        damaged_image = apply_motion_blur(damaged_image, blur_kernel)

    # Edge Distortion
    if st.sidebar.checkbox("Apply Edge Distortion"):
        operation = st.sidebar.radio("Operation", ["Erosion", "Dilation"])
        kernel_size = st.sidebar.slider("Kernel Size", 3, 15, 5)
        iterations = st.sidebar.slider("Iterations", 1, 10, 1)
        damaged_image = apply_edge_distortion(
            damaged_image, operation=operation.lower(), kernel_size=kernel_size, iterations=iterations
        )

    # Vignetting
    if st.sidebar.checkbox("Apply Vignetting Effect"):
        vignette_intensity = st.sidebar.slider("Vignette Intensity", 0.0, 1.0, 0.5)
        damaged_image = apply_vignetting(damaged_image, vignette_intensity)

    # Display and Download
    st.image(damaged_image, channels="BGR", caption="Damaged Image")
     # Download button
    result_image = Image.fromarray(cv2.cvtColor(damaged_image, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    result_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    # Create a new file name by adding " Damaged" to the original name
    damaged_file_name = f"{original_name} Damaged{original_extension}"
    
    st.download_button(
        label="Download Damaged Image",
        data=byte_im,
        file_name=damaged_file_name,
        mime="image/png",
    )
