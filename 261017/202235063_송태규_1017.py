# import libraries for executing code
import numpy as np
import matplotlib.pyplot as plt
import cv2

# TODO: Morphological operation을 적용하여 최종 binary mask 생성
# TODO: Binary mask로 외곽선 검출
# TODO: 최종 농구공 영역 및 외곽선 선택


lower_basketball_lab = np.array([70, 130, 70]) # lower bound for basketball in LAB color space
upper_basketball_lab = np.array([180, 230, 185]) # upper bound for basketball in LAB color space

basketball = cv2.imread('../src/images/basketball_crop.jpg') # relative directory

# Convert BGR to LAB
basketball_lab = cv2.cvtColor(basketball, cv2.COLOR_BGR2LAB)

# Split each channel
l, a, b = cv2.split(basketball_lab)

# Generate CLAHE (Set clipLimit 3.0, tileGridSize 8x8)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# Generate enhanced image by applying CLAHE to the L channel
l_clahe_enhanced = clahe.apply(l)
basketball_enhanced_lab = cv2.merge((l_clahe_enhanced, a, b)) # Merge enhanced L, a, b channels to get the enhanced LAB image

# Generate binary mask by enhanced image
mask_enhanced = cv2.inRange(basketball_enhanced_lab, lower_basketball_lab, upper_basketball_lab)

# Convert enhanced LAB image to RGB for visualization
basketball_enhanced_rgb = cv2.cvtColor(basketball_enhanced_lab, cv2.COLOR_LAB2RGB)

# Create a circular kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) 

# Apply morphological operation and generate final binary mask 
# Opening operation 3 times to remove noise and small objects
mask_enhanced_opened3 = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel, iterations=3)

final_basketball_mask = mask_enhanced_opened3 # final mask

final_img = basketball_enhanced_rgb.copy() # final image

contours, _ = cv2.findContours(
    final_basketball_mask,        # binary mask
    cv2.RETR_EXTERNAL,       # detect only the outer edges of the object with holes
    cv2.CHAIN_APPROX_SIMPLE  # store only the corner coordinates instead of all pixel coordinates along a straight path
)

if len(contours) > 0:
    # The largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Generate the largest contour only filled binary mask
    final_mask = np.zeros_like(final_basketball_mask)
    cv2.drawContours(
        final_mask,
        [largest_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # The largest contour only filled binary mask to overlay with a semi-transparent green color
    mask_color = np.zeros_like(final_img)
    mask_color[:, :, 1] = final_mask
    alpha = 0.35
    final_img = cv2.addWeighted(final_img, 1.0, mask_color, alpha, 0)

    # Draw the largest contour (green outline)
    cv2.drawContours(final_img, [largest_contour], -1, (0,255,0), 2)
    
plt.figure(figsize=(16,8)) # Visualization
plt.subplot(1,3,1), plt.imshow(mask_enhanced, cmap="gray"), plt.title("Enhanced Mask")
plt.subplot(1,3,2), plt.imshow(final_basketball_mask, cmap="gray"), plt.title("Opening 3 times")
plt.subplot(1,3,3), plt.imshow(final_img), plt.title("Largest Contour (Basketball)")


for ax in plt.gcf().axes: # Turn off axis for all subplots
    ax.axis("off")

plt.show()
