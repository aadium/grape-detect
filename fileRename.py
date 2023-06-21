import os

folder_path = 'test'

# Get the list of files in the folder
file_names = os.listdir(folder_path)

# Separate the files into JPG and text files
jpg_files = [file for file in file_names if file.lower().endswith(('.jpg', '.jpeg'))]
txt_files = [file for file in file_names if file.lower().endswith('.txt')]

# Sort the file lists alphabetically
jpg_files.sort()
txt_files.sort()

# Rename the files
for i, (jpg_file, txt_file) in enumerate(zip(jpg_files, txt_files), 1):
    # Rename JPG file
    jpg_file_new = f'test_{i}.jpg'
    jpg_file_path = os.path.join(folder_path, jpg_file)
    jpg_file_new_path = os.path.join(folder_path, jpg_file_new)
    os.rename(jpg_file_path, jpg_file_new_path)
    print(f'Renamed "{jpg_file}" to "{jpg_file_new}"')

    # Rename text file
    txt_file_new = f'test_{i}.txt'
    txt_file_path = os.path.join(folder_path, txt_file)
    txt_file_new_path = os.path.join(folder_path, txt_file_new)
    os.rename(txt_file_path, txt_file_new_path)
    print(f'Renamed "{txt_file}" to "{txt_file_new}"')