import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import unetpp
import numpy as np
from skimage import measure, morphology

# Assuming 'model' is your trained model
model = unetpp.UNetPlusPlus(n_channels=1, n_classes=1, bilinear=False, deep_supervision=True)
model.load_state_dict(torch.load('checkpoints/unetpp.pth'))

# Load the image
image = Image.open('Datasets/TiO2/test/img/1908288.tif').convert('L')
transform = transforms.Compose([
    transforms.Resize((768,1024)),  # Resize if your model was trained on a different size
    transforms.ToTensor()
])

# Prepare the image for the model
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Set the model to evaluation mode
model.eval()

# Predict
with torch.no_grad():
    output = model(input_tensor)

# Assuming the output is a segmentation mask
predicted_prob = output[0].squeeze()  # Remove batch dimension if present

# #show the image
# plt.imshow(predicted_prob, cmap='gray')
# plt.show()

# # Apply a threshold to convert to binary image, if necessary
binary_mask = predicted_prob.numpy() > 0.5

# plt.imshow(binary_mask, cmap='gray')
# plt.show()

# Label connected components
labeled_mask = measure.label(binary_mask)
particles = measure.regionprops(labeled_mask)
print(labeled_mask)
print(type(labeled_mask))
print(labeled_mask.shape)
diameters = [prop.equivalent_diameter for prop in particles]

# Optionally apply morphological operations to clean up the mask
cleaned_mask = morphology.remove_small_objects(labeled_mask, min_size=100)  # Adjust min_size as needed
print(type(particles), len(particles))
print(f'Number of particles: {len(np.unique(cleaned_mask)) - 1}')
print(len(cleaned_mask))

def filter_border_particles(labeled_mask):
    border_labels = np.unique(np.concatenate((labeled_mask[0, :], labeled_mask[-1, :], labeled_mask[:, 0], labeled_mask[:, -1])))
    return np.isin(labeled_mask, border_labels, invert=True) * labeled_mask
# Apply filtering
filtered_mask = filter_border_particles(cleaned_mask)

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image, cmap='gray')  # Original grayscale image
ax.imshow(filtered_mask, alpha=0.5, cmap='jet')  # Overlay filtered mask
for particle in particles:
    y, x = particle.centroid
    ax.text(x, y, f'{particle.equivalent_diameter:.2f}', color='white')
ax.set_title('Segmented Particles with Diameters')
ax.axis('off')
plt.show()