# import cv2
# import numpy as np

# There are 2_893_401 lines in input
# There were 2_893_402 lines in output ( last line is empty )
# => But output file was expecting only 319_225



# def reconstruct_smoothed_image(outputs, image_shape, kernel_shape, padding_method):
#     """
#     Reconstructs the smoothed image using provided multiplication outputs.

#     Args:
#         outputs (list): List of integers representing the multiplication outputs.
#         image_shape (tuple): Original image shape (height, width).
#         kernel_shape (tuple): Kernel shape (height, width).
#         padding_method (str): Padding method used in image smoothing.

#     Returns:
#         numpy.ndarray: The smoothed image as a NumPy array.
#     """

#     # Calculate padding size for each dimension based on padding method
#     kernel_center = (kernel_shape[0] - 1) // 2
#     pad_top, pad_bottom = kernel_center, kernel_center
#     pad_left, pad_right = kernel_center, kernel_center


#     if padding_method == "reflect":
#         valid_height, valid_width = image_shape[0] - 2 * pad_top, image_shape[1] - 2 * pad_left

#     elif padding_method == "replicate":
#         valid_height, valid_width = image_shape[0] - 2 * pad_top, image_shape[1] - 2 * pad_left

#     else:
#         raise ValueError("Invalid padding method. Choose 'reflect' or 'replicate'.")

#     print(valid_height,valid_width,valid_width*valid_height)
#     expected_outputs = valid_height * valid_width
#     if len(outputs)-1 != expected_outputs:
#         raise ValueError("Number of outputs does not match expected number of pixels.")
#     print("Number of outputs mathches expected number of pixels !!")

#     # Initialize smoothed image with zeros (adjust data type if needed)
#     smoothed_image = np.zeros(image_shape, dtype=np.float32)  # Adjust data type as needed

#     # Iterate over the valid region of the image (excluding padded areas)
#     for y in range(kernel_center, image_shape[0] - kernel_center):
#         for x in range(kernel_center, image_shape[1] - kernel_center):
#             # Calculate offset for current pixel in the output list
#             offset = (y  - kernel_center) * valid_width + (x - kernel_center)

#             # Extract corresponding output value
#             output_value = outputs[offset]

#             # Normalize output value based on kernel sum (optional for averaging)
#             # kernel_sum = np.sum(kernel)  # Uncomment if using averaging kernel
#             # smoothed_image[y, x] = output_value / kernel_sum  # Uncomment for averaging

#             # Directly assign output value to smoothed image (adjust based on your kernel)
#             smoothed_image[y, x] = output_value

#     return smoothed_image

# def main():
#     with open("./output_from_multiplier.dat", "r") as f:
#         outputs = [int(line.strip()) for line in f.readlines()]

#     image_shape = (567, 567) 
#     kernel_shape = (3, 3) 
#     padding_method = "reflect" 

#     smoothed_image = reconstruct_smoothed_image(outputs, image_shape, kernel_shape, padding_method)
#     cv2.imwrite("smoothed_image.jpg", smoothed_image.astype(np.uint8))


# if __name__ == "__main__":
# 
#   main()


import cv2
import numpy as np

def reconstruct_smoothed_image(outputs, image_shape, kernel_shape, padding_method):
    """
    Reconstructs the smoothed image using provided multiplication outputs.

    Args:
        outputs (list): List of integers representing the multiplication outputs.
        image_shape (tuple): Original image shape (height, width).
        kernel_shape (tuple): Kernel shape (height, width).
        padding_method (str): Padding method used in image smoothing.

    Returns:
        numpy.ndarray: The smoothed image as a NumPy array.
    """

    # Calculate padding size for each dimension based on padding method
    kernel_center = (kernel_shape[0] - 1) // 2
    pad_top, pad_bottom = kernel_center, kernel_center
    pad_left, pad_right = kernel_center, kernel_center

    # Calculate the valid region of the image (excluding padded areas)
    valid_height, valid_width = image_shape[0] - 2 * pad_top, image_shape[1] - 2 * pad_left
    expected_outputs = valid_height * valid_width * 9

    # print(len(outputs))
    # print(expected_outputs)
    # if len(outputs) != expected_outputs:
    #     raise ValueError("Number of outputs does not match the expected number based on kernel size.")

    # Initialize smoothed image
    smoothed_image = np.zeros((valid_height, valid_width), dtype=np.float32)

    output_index = 0

    # Iterate over the valid region of the image (excluding padded areas)
    for y in range(valid_height):
        for x in range(valid_width):
            # Sum the 9 outputs (one for each element in the 3x3 kernel)
            sum_value = 0
            for i in range(9):
                sum_value += outputs[output_index]
                output_index += 1

            # Assign the sum to the smoothed pixel
            smoothed_image[y, x] = sum_value

    return smoothed_image

def main():
    with open("./output_from_multiplier.dat", "r") as f:
        outputs = [int(line.strip()) for line in f.readlines()]

    image_shape = (567, 567)
    kernel_shape = (3, 3)
    padding_method = "reflect"

    smoothed_image = reconstruct_smoothed_image(outputs, image_shape, kernel_shape, padding_method)
    cv2.imwrite("smoothed_image.jpg", smoothed_image.astype(np.uint8))

if __name__ == "__main__":
    main()
