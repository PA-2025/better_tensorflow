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