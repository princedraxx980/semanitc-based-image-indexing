📸 Semantic-Based Image Indexing using ResNet50 and CIFAR-10
This project implements a semantic-based image indexing system using a pretrained ResNet50 model on the CIFAR-10 dataset. The goal is to enable image retrieval based on semantic similarity rather than just pixel-level similarity, enabling more meaningful and human-like search results.

📂 Table of Contents
About
Tech Stack
Dataset
Model Architecture
Installation
Usage
Results
Future Work
Contributors
License

📌 About
Traditional image indexing techniques often rely on low-level features such as color and texture. This project takes a semantic approach by leveraging deep features extracted from ResNet50 to represent and compare images based on their content.

🛠 Tech Stack
Python
TensorFlow / Keras
NumPy
Matplotlib
Scikit-learn
ResNet50 (Pretrained on ImageNet)

📊 Dataset
CIFAR-10: 60,000 32x32 color images in 10 classes, with 6,000 images per class.
Classes include: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

🧠 Model Architecture
ResNet50 is used as a feature extractor (excluding top layers).
Features are extracted from intermediate layers to obtain semantically rich image embeddings.
Cosine similarity is used to compute similarity between image embeddings for indexing and retrieval.

🔧 Installation
1. Clone the repository: git clone https://github.com/yourusername/semantic-image-indexing.git
   cd semantic-image-indexing.
2. Install required packages: pip install -r requirements.txt
3. Optional) Set up a virtual environment for isolation.

▶️ Usage
Run the main script: python semantic_image_indexing.py

This will:
  1. Load the CIFAR-10 dataset
  2. Preprocess and pass images through ResNet50
  3. Extract feature vectors
  4. Store vectors in a searchable format
  5. Allow querying by displaying top-N semantically similar images

📈 Results
  1. Effective clustering of semantically similar images
  2. Real-time retrieval based on feature similarity
  3. Visual plots showing retrieved image sets

🚀 Future Work
  1. Use deeper embeddings from ViT or CLIP
  2. Support larger and more complex datasets (e.g., ImageNet)
  3. Add a web interface for image upload and search
  4. Explore dimensionality reduction (e.g., PCA/TSNE) for visualization


👥 Contributors
Drushya Bhaisare

 
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
