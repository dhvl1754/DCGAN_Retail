import tensorflow as tf 
import os

#path to the datset folders
Training_path = "data/fruits-360/Training"
Testing_path = "data/fruits-360/Testing"

img_height, img_width = 64,64
BATCH_SIZE = 32

def load_and_process_images(dataset_path):
    dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        label_mode = None,
        image_size = (img_height, img_width),
        batch_size = BATCH_SIZE
    )

    #Normalize images to [-1,1]

    dataset = dataset.map(lambda x: (x/127.5) -1 )

    return dataset

if __name__ == "__main__":
    training_dataset = load_and_process_images(Training_path)
    print(f"Training dataset loaded with {len(training_dataset)} batches.")