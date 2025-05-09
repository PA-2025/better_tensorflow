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

## Add dataset

- check dataet version on `tools/scrapper/README.md`
- download dataset
- unzip dataset in `python/api/data/`