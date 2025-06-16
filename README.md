# Build lib

---

```bash
maturin build && pip install .
```

# Launch api

---

```bash
pip install -r python/requirements.txt
fastapi dev python/api/app.py
```

### Swagger on :

- http://127.0.0.1:8000/docs

### Debug rust :

```bash
export RUST_BACKTRACE=1
```

# Add dataset

---

- check dataet version on `tools/scrapper/README.md`
- download dataset
- unzip dataset in `python/api/data/`

- to convert dataset soung to png you can use the script `tools/soung_to_image.py dataset_path_img dataset_path_soung`


# Launch tests in csv

- Modify this var to define layer
```python
self.test_layers = [[4], [4, 4], [10], [4, 10]]
```

```bash
export PYTHONPATH=$(pwd)/src
py tools/test_better_model.py 4 10000
```


# Convert dataset to mongo and use it

---

```bash
cp .env.exemple .env
```
```bash
docker compose up -d
```
```bash
py tools/import_dataset_to_mongo.py python/api/data/music_spec/
```
#### to use it in api
```bash
export USE_MONGO=1
```
