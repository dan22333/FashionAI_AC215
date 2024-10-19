import os
import shutil

# Names of your original two images
image1 = 'img0.jpg'  # Replace with the actual filename
image2 = 'img1.jpg'  # Replace with the actual filename

# List to hold the two images
images = [image1, image2]

# Loop to create 100 images
for i in range(2, 100):
    # Decide which image to duplicate, alternating between the two
    image_to_copy = images[i % 2]
    
    # Create the new image name
    new_image_name = f'img{i}.jpg'
    
    # Copy and rename the image
    shutil.copy(image_to_copy, new_image_name)

print("Images duplicated and renamed successfully!")
