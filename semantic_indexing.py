import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset to the program
def load_cifar10():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images = train_images / 255.0  # Normalize images
    test_images = test_images / 255.0   # Normalize images
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    return train_images, train_labels, test_images, test_labels, class_names

# Extract features using ResNet50 pre trained model
def extract_features(images, model):
    images_resized = tf.image.resize(images, (224, 224))  # Resize to ResNet50 input size
    images_preprocessed = preprocess_input(images_resized)
    features = model.predict(images_preprocessed, verbose=0)
    return features

# Visualize images
def visualize_images(images, labels, class_names, num_images=5):
    plt.figure(figsize=(10, 2))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i])
        plt.title(class_names[labels[i][0]])
        plt.axis('off')
    plt.show()

# Find similar images
def find_similar_images(query_features, dataset_features, dataset_images, num_results=5):
    similarities = cosine_similarity(query_features, dataset_features)
    similar_indices = np.argsort(similarities[0])[::-1][:num_results]
    return [dataset_images[i] for i in similar_indices]

# Main program
if __name__ == "__main__":
    # Load CIFAR-10 dataset
    train_images, train_labels, test_images, test_labels, class_names = load_cifar10()
    print("Dataset loaded. Train images shape:", train_images.shape)

    # Initialize ResNet50 model
    resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    # Extract features from CIFAR-10 dataset
    print("Extracting features for training images...")
    train_features = extract_features(train_images[:1000], resnet_model) 
    print("Feature extraction completed.")

    # Index dataset images
    print("Indexing dataset images...")
    dataset_features = train_features
    dataset_images = train_images[:1000]

    # Select a random query image(Temproray)
    random_idx = np.random.randint(0, test_images.shape[0])  # Random index from test images
    query_image = test_images[random_idx]  # Use the random image as a query, temprory would develop an input query image later on.
    query_image_preprocessed = preprocess_input(tf.image.resize(query_image, (224, 224))[None, ...])
    query_features = resnet_model.predict(query_image_preprocessed, verbose=0)

    # Retrieve similar images
    print("Finding similar images...")
    similar_images = find_similar_images(query_features, dataset_features, dataset_images, num_results=5)

    # Visualize the query image and similar images
    print("Displaying query image and similar images...")
    plt.figure(figsize=(10, 2))
    plt.subplot(1, 6, 1)
    plt.imshow(query_image)
    plt.title("Query Image")
    plt.axis('off')

    for i, similar_image in enumerate(similar_images):
        plt.subplot(1, 6, i + 2)
        plt.imshow(similar_image)
        plt.title(f"Match {i + 1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
