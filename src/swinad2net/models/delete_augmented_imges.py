import os 

def delete_augmented_images(root='./data'):
    for dir_path, _, files in os.walk(root):
        for file_name in files:
            if "aug" in file_name:
                try:
                    os.remove(os.path.join(dir_path, file_name))
                except Exception as e:
                    print(f"Error deleting file {file_name}: {e}")

if __name__ == "__main__":
    delete_augmented_images(root='./data')
                    