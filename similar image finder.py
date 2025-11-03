import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from google.colab import files

class ImageSimilarityFinder:
    def __init__(self, model_type='resnet50', feature_layer='avg_pool', similarity_threshold=0.5):
        self.model_type = model_type
        self.feature_layer = feature_layer
        self.similarity_threshold = similarity_threshold
        self.feature_extractor = None
        self.nearest_neighbors = None
        self.features_database = None
        self.image_paths = None

        self._setup_model()

    def _setup_model(self):
        try:
            base_model = ResNet50(weights='imagenet', include_top=True)
            self.feature_extractor = tf.keras.Model(
                inputs=base_model.input,
                outputs=base_model.get_layer(self.feature_layer).output
            )
        except Exception as e:
            print(f"Error setting up model: {str(e)}")
            raise

    def _load_and_preprocess_image(self, image_path, target_size=(224, 224)):
        try:
            img = image.load_img(image_path, target_size=target_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            return img_array
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None

    def build_database(self, image_directory, batch_size=32):
        if not os.path.exists(image_directory):
            raise ValueError(f"Directory not found: {image_directory}")

        image_paths = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

        if not image_paths:
            raise ValueError(f"No valid images found in directory: {image_directory}")

        self.image_paths = image_paths
        features_list = []

        print(f"Processing {len(image_paths)} images...")

        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []

            for img_path in batch_paths:
                img_array = self._load_and_preprocess_image(img_path)
                if img_array is not None:
                    batch_images.append(img_array[0])

            if batch_images:
                batch_images = np.array(batch_images)
                batch_features = self.feature_extractor.predict(batch_images, verbose=0)
                batch_features = batch_features / np.linalg.norm(batch_features, axis=1)[:, np.newaxis]
                features_list.extend(batch_features)

        if not features_list:
            raise ValueError("No features could be extracted from the images")

        self.features_database = np.array(features_list)

        n_neighbors = min(max(1, len(self.features_database)), 10)
        self.nearest_neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        self.nearest_neighbors.fit(self.features_database)

    def find_similar_images(self, query_image_path, num_results=5):
        if self.features_database is None or self.nearest_neighbors is None:
            raise ValueError("Database not built. Call build_database() first.")

        query_features = self.extract_features(query_image_path)

        if query_features is None:
            raise ValueError(f"Could not process query image: {query_image_path}")

        n_neighbors = min(num_results + 1, len(self.features_database))
        if n_neighbors < 1:
            raise ValueError("Not enough images in database to find similar ones")

        distances, indices = self.nearest_neighbors.kneighbors(query_features, n_neighbors=n_neighbors)

        similar_images = {'similar': [], 'unique': []}
        for idx, distance in zip(indices[0][1:], distances[0][1:]):
            if distance < self.similarity_threshold:
                similar_images['similar'].append({
                    'path': self.image_paths[idx],
                    'distance': distance
                })
            else:
                similar_images['unique'].append({
                    'path': self.image_paths[idx],
                    'distance': distance
                })

        return similar_images

    def extract_features(self, image_path):
        img_array = self._load_and_preprocess_image(image_path)
        if img_array is not None:
            features = self.feature_extractor.predict(img_array, verbose=0)
            normalized_features = features / np.linalg.norm(features)
            return normalized_features
        return None

    def visualize_results(self, query_image_path, similar_images):
        if not (similar_images['similar'] or similar_images['unique']):
            print("No similar or unique images to visualize")
            return

        total_images = len(similar_images['similar']) + len(similar_images['unique']) + 1
        fig = plt.figure(figsize=(15, 5))

        plt.subplot(1, total_images, 1)
        query_img = Image.open(query_image_path)
        plt.imshow(query_img)
        plt.title('Query Image')
        plt.axis('off')

        count = 2
        for category, images in similar_images.items():
            for img_data in images:
                plt.subplot(1, total_images, count)
                similar_img = Image.open(img_data['path'])
                plt.imshow(similar_img)
                plt.title(f"{category.capitalize()}\nDistance: {img_data['distance']:.2f}")
                plt.axis('off')
                count += 1

        plt.tight_layout()
        plt.show()

def demo_image_similarity_finder():
    try:
        finder = ImageSimilarityFinder(similarity_threshold=0.5)

        uploaded = files.upload()

        image_directory = 'uploaded_images'
        if not os.path.exists(image_directory):
            os.makedirs(image_directory)

        for filename in uploaded.keys():
            file_path = os.path.join(image_directory, filename)
            with open(file_path, 'wb') as f:
                f.write(uploaded[filename])

        print(f"Building database from: {image_directory}")
        finder.build_database(image_directory)

        image_files = [f for f in os.listdir(image_directory)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

        if not image_files:
            print("No images found in the directory. Please upload some images and try again.")
            return

        query_image = os.path.join(image_directory, image_files[0])
        print(f"Using query image: {query_image}")

        similar_images = finder.find_similar_images(query_image, num_results=5)

        finder.visualize_results(query_image, similar_images)

    except Exception as e:
        print(f"Error in demo: {str(e)}")

# Run the demo
demo_image_similarity_finder()
