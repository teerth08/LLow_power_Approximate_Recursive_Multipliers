import cv2
import numpy as np

input_multiply_file = "./input_to_multiply.dat"
output_multiply_file = "./output_from_multiplier.dat"

def log_multiply(a, b):
    with open(input_multiply_file, "a") as f:
        f.write(f"{a} {b}\n")



# Function to apply a filter and log multiplications
def apply_filter_with_verilog(image, kernel):

    img_h, img_w = image.shape
    kernel_h, kernel_w = kernel.shape

    # Padding the image to handle border issues
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    padded_image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)
    
    # Placeholder for result
    result_image = np.zeros_like(image)
    
    # Perform convolution manually (and log multiplications)
    for i in range(img_h):
        for j in range(img_w):
            conv_sum = 0
            for ki in range(kernel_h):
                for kj in range(kernel_w):
                    pixel_value = padded_image[i + ki, j + kj]
                    kernel_value = kernel[ki, kj]
                    log_multiply(pixel_value, kernel_value)  # Log multiplication inputs
                    conv_sum += pixel_value * kernel_value  # Actual multiplication done by Verilog

            result_image[i, j] = min(max(conv_sum, 0), 255)  # Ensure pixel values are in [0, 255]
    
    # After logging all inputs, let's assume the verilog has processed and given results.
    # We'll read the output from the file.
    with open(output_multiply_file, "r") as f:
        lines = f.readlines()
    
    # Use the Verilog outputs to adjust the final values (simulating Verilog usage)
    verilog_outputs = [int(line.strip()) for line in lines]
    idx = 0
    for i in range(img_h):
        for j in range(img_w):
            result_image[i, j] = verilog_outputs[idx]  # Replace the computed value with Verilog result
            idx += 1
    
    return result_image

# Load the image
image = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)

# Define a 3x3 low-pass (smoothing) kernel
low_pass_kernel = np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]], dtype=np.float32) / 9.0

# Define a 3x3 high-pass (sharpening) kernel
high_pass_kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]], dtype=np.float32)

# Perform low-pass filtering (smoothing)
smoothed_image = apply_filter_with_verilog(image, low_pass_kernel)

# Save the smoothed image
cv2.imwrite('smoothed_image.jpg', smoothed_image)
sharpened_image = apply_filter_with_verilog(image, high_pass_kernel)

# Save the sharpened image
cv2.imwrite('sharpened_image.jpg', sharpened_image)
print("Smoothing and sharpening complete!")