import os

def get_image_paths(folder_path):
    image_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)
    return image_paths

# Folder paths
test_folder_path = 'test'
train_folder_path = 'train'

# Get image paths for test and train folders
test_image_paths = get_image_paths(test_folder_path)
train_image_paths = get_image_paths(train_folder_path)

# Save image paths to a text file
with open('image_paths.txt', 'w') as f:
    f.write("Test Folder Images:\n")
    for path in test_image_paths:
        f.write(path + '\n')
    
    f.write("\nTrain Folder Images:\n")
    for path in train_image_paths:
        f.write(path + '\n')

print("Image paths saved to image_paths.txt file.")