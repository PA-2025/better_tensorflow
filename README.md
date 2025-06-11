# Build lib

---

```bash
maturin build && pip install .
```

# Launch api

---

```bash
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
export PYTHONPATH=$(pwd)
py tools/test_better_model.py 4 10000
```