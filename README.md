# 🎵 Music Classification Project

[![Rust](https://img.shields.io/badge/Rust-🦀-orange)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-🚀-green)](https://fastapi.tiangolo.com/)

A hybrid Rust + Python project for classifying music based on spectrograms using a FastAPI backend.

---

## 🛠️ 1. Build the Rust Library

```bash
maturin build && pip install .
```

---

## 🚀 2. Launch the FastAPI Server

### Install Python dependencies:

```bash
pip install -r python/requirements.txt
```

### Start the server:

```bash
fastapi dev python/api/app.py
```

### 📘️ Swagger UI:

* [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🐛 3. Debugging Rust Code

Enable detailed Rust error messages:

```bash
export RUST_BACKTRACE=1
```

---

## 🎿 4. Add a Dataset

1. Check dataset version in `tools/scrapper/README.md`
2. Download and unzip the dataset into:

```bash
python/api/data/
```

3. Convert audio files to PNG spectrograms:

```bash
python tools/soung_to_image.py <output_img_path> <input_audio_path>
```

---

## ✅ 5. Run CSV-Based Model Tests

1. Edit layer configurations in the script:

```python
self.test_layers = [[4], [4, 4], [10], [4, 10]]
```

2. Run the tests:

```bash
export PYTHONPATH=$(pwd)/src
python tools/test_better_model.py 4 10000
```

---

## 📦 6. Convert Dataset to MongoDB & Use It

1. Copy the environment file:

```bash
cp .env.exemple .env
```

2. Launch MongoDB with Docker:

```bash
docker compose up -d
```

3. Import the dataset:

```bash
python tools/import_dataset_to_mongo.py python/api/data/music_spec/
```

4. Enable Mongo usage in the API:

```bash
export USE_MONGO=1
```

---


