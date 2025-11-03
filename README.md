# similar-image-finder-using-deep-learning-

# Image Similarity Finder using ResNet50 & Nearest Neighbors

A **lightweight image similarity search system** built with **TensorFlow**, **ResNet50**, and **scikit-learn's Nearest Neighbors**. This tool extracts deep features from images using a pre-trained ResNet50 model and finds visually similar (or unique) images using cosine similarity.

Perfect for:
- Duplicate image detection
- Visual search engines
- Image clustering
- Content-based image retrieval (CBIR)

---

## Features

- Uses **ResNet50 (ImageNet pre-trained)** for robust feature extraction
- **Cosine similarity** via `NearestNeighbors`
- Batch processing with progress bar (`tqdm`)
- Visualizes query image + similar/unique results
- Runs directly in **Google Colab** with file upload support
- Configurable similarity threshold

---

## How It Works

1. **Feature Extraction**: Images are preprocessed and passed through ResNet50 (global average pooling layer).
2. **Normalization**: Feature vectors are L2-normalized.
3. **Similarity Search**: Nearest neighbors are found using cosine distance.
4. **Classification**: Images below threshold → `similar`, above → `unique`.
5. **Visualization**: Results displayed in a clean grid.

---

## Requirements

```txt
tensorflow>=2.10
numpy
scikit-learn
matplotlib
Pillow
tqdm
```

> Designed to run in **Google Colab** (uses `google.colab.files`)

---

## Usage (Google Colab)

1. Open in [Google Colab](https://colab.research.google.com)
2. Run all cells
3. Upload your images when prompted
4. The first uploaded image is used as the **query**
5. View similar and unique matches with distances

```python
finder = ImageSimilarityFinder(similarity_threshold=0.5)
finder.build_database("your_image_folder/")
results = finder.find_similar_images("query.jpg", num_results=5)
finder.visualize_results("query.jpg", results)
```

---

## Example Output

```
Query Image       | Similar (0.12) | Similar (0.25) | Unique (0.68) | Unique (0.89)
```

> Distances closer to `0` = more similar

---

## Customization

| Parameter | Description | Default |
|--------|-------------|---------|
| `model_type` | Base model (only ResNet50 supported now) | `'resnet50'` |
| `feature_layer` | Layer to extract features from | `'avg_pool'` |
| `similarity_threshold` | Max cosine distance for "similar" | `0.5` |
| `num_results` | Number of similar/unique images to return | `5` |

---

## Project Structure

```
├── ImageSimilarityFinder.ipynb     # Main notebook (your code)
├── README.md                       # This file
├── uploaded_images/                # Created at runtime (Colab)
└── requirements.txt                # (Optional) pip dependencies
```

---

## Local Installation (Optional)

If running locally:

```bash
pip install tensorflow scikit-learn matplotlib pillow tqdm
```

Then modify `demo_image_similarity_finder()` to point to a local directory instead of `files.upload()`.

---

## Future Improvements

- [ ] Support for custom models (MobileNet, EfficientNet)
- [ ] Web UI with Streamlit/Gradio
- [ ] Save/load feature database
- [ ] CLI interface
- [ ] Duplicate grouping & removal

---

## License

[MIT License](LICENSE) – Free to use, modify, and distribute.

---

## Author

**Your Name**  
🔗 [GitHub Profile](https://github.com/saikotimedabalimi)  
📧 medabalimisaikotil@gmail.com

---

⭐ **Star this repo if you found it useful!**
```

---

### Optional: Add `requirements.txt`

Create a file named `requirements.txt` in your repo:

```txt
tensorflow>=2.10
numpy
scikit-learn
matplotlib
Pillow
tqdm
```

---

### Optional: Add License

Create `LICENSE` file with MIT license:

```txt
MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge...
```

---

Let me know if you'd like a **Streamlit version**, **CLI tool**, or **Docker support** next!
```
