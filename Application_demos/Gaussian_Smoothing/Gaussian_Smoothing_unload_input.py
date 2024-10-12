import cv2
import numpy as np


def generate_multiplication_inputs(image, kernel, padding_method="reflect"):
  """
  Generates multiplication inputs required for image smoothing using a Verilog multiplier.
  Args:
      image (numpy.ndarray): Input image as a NumPy array.
      kernel (numpy.ndarray): Smoothing kernel as a NumPy array.
      padding_method (str, optional): Padding method for image borders.
          Defaults to "reflect".

  Returns:
      list: List of lists, where each inner list contains two integers representing
          the pixel values to be multiplied for a specific output pixel.
  """

  # Calculate padding size for each dimension
  kernel_center = (kernel.shape[0] - 1) // 2
  pad_top, pad_bottom = kernel_center, kernel_center
  pad_left, pad_right = kernel_center, kernel_center

  # Pad the image if necessary
  if padding_method == "reflect":
    image = cv2.copyMakeBorder(image, top=pad_top, bottom=pad_bottom,
                              left=pad_left, right=pad_right, borderType=cv2.BORDER_REFLECT)
  elif padding_method == "replicate":
    image = cv2.copyMakeBorder(image, top=pad_top, bottom=pad_bottom,
                              left=pad_left, right=pad_right, borderType=cv2.BORDER_REPLICATE)
  else:
    raise ValueError("Invalid padding method. Choose 'reflect' or 'replicate'.")

  # Iterate over image and generate multiplication inputs
  multiplication_inputs = []
  height, width = image.shape[:2]

  for y in range(kernel_center, height - kernel_center):
    for x in range(kernel_center, width - kernel_center):

      # Extract pixel neighborhood
      patch = image[y - kernel_center:y + kernel_center + 1,
                    x - kernel_center:x + kernel_center + 1]

      # Flatten patch and kernel for element-wise multiplication
      patch = patch.flatten()
      kernel = kernel.flatten()

      # Generate multiplication inputs (pixel value * kernel weight)
      for pixel_value, weight in zip(patch, kernel):
        multiplication_inputs.append([int(pixel_value), int(weight)])

  return multiplication_inputs




def main():
    image = cv2.imread("./Lena_Original_Image__512x512-pixels_W640.jpg", cv2.IMREAD_GRAYSCALE)
    
    # Correct kernal to be compatible with 8x8 multiplier [Mentioned in paper]
    kernel = np.array(
        [[97, 121, 97], 
         [121, 151, 121], 
         [97, 121, 97]]
    )

    multiplication_inputs = generate_multiplication_inputs(image, kernel)

    with open("./input_to_multiply.dat", "w") as f:
        for input1, input2 in multiplication_inputs:
            f.write(f"{input1} {input2}\n")

    print("Inputs for Verilog multiplier generated successfully!")

if __name__ == "__main__":
  main()

  

'''
prompts

[Worked]
I have a verilog multiplier, in order to test how good it is, I want help.
I need python code to perform image smoothing. But it's going to in 2 files of code :-

( dump_inputs_to_multiply.py )
1. Takes in an image and prints the inputs, every time you need to multiply 2 numbers onto a file called inputs_for_verilog_multiplier.dat
 The processing of each pixel requires a number of multiplications that depends on the size of the kernel. In fact, the value of the 
 modified pixel is the weighted average of the neighboring pixels. While doing this just print all the inputs you need to multiply onto the file

( read_outputs_and_complete_image_smoothing.py )
 2. Reads the output of the multiplication FROM A FILE outputs_from_verilog_multiplier.dat ; and uses it to construct the smoothened out image.
 
 No where does python multiply anything, all the outputs are read from a file. ( each line corresponds to the multiplication output of the inputs you feed in the first first file )


[prompt_2]
I have a file called inputs_for_verilog_multiplier.dat './test_input.dat'  where each line has 2 numbers.
I also have a mutiplier module n8_5

Write verilog code that takes input from each line of inputs_for_verilog_multiplier.dat and uses the module to multiply them and and the output needs to printed onto
a file called './test_output.dat'
'''