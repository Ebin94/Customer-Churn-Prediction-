#  Technical Details & MLOps Architecture

This document provides a deep-dive into the engineering decisions and patterns used to make this ML system robust, reproducible, and ready for production.

---

## 1.  Unified Scikit-Learn Pipeline (`src/train.py`)
**Problem**: Training data preparation and inference data preparation often drift apart over time ("training-serving skew"). If a model is trained using a specific set of feature scalers or encoders, the exact same transformations must be applied to the inference payload in production.

**Solution**: Rather than saving individual components (e.g., `scaler.pkl`, `encoder.pkl`, `model.pkl`) and manually reconstructing the logic in the API, we use a single scikit-learn `Pipeline` and `ColumnTransformer`.
- **Consistency**: The `Pipeline` bundles missing values imputation, scaling, one-hot encoding, and the final estimator into a unified binary object (`churn_pipeline.pkl`).
- **Safety**: Calling `pipeline.predict(X)` automatically handles all underlying transformations correctly. Code inside the API is minimal, eliminating the chance of human error during feature engineering replication.

---

## 2. Fast Model Loading with FastAPI Lifespan (`api/main.py`)
**Problem**: Loading a machine learning model into memory horizontally during an API request generates massive latency and defeats the purpose of an asynchronous web framework.

**Solution**: We utilize FastAPI's built-in `@asynccontextmanager` (`lifespan`) to load the trained model pipeline into memory **exactly once** during the application's startup phase. 
- **Efficiency**: All subsequent `/predict` requests utilize the preloaded, global `model_pipeline` object, allowing predictions to execute in milliseconds. 
- **Graceful degradation**: The lifespan handler includes robust error catching. If the model fails to load, the API doesn't crash; it still boots up, surfaces the failure securely on the `/health` endpoint, and alerts connected orchestration tools.

---

## 3.  Strict Payload Validation with Pydantic
**Problem**: Machine learning models will often throw obscure exceptions (e.g., `ValueError` for dimension mismatch or invalid string casts) if the incoming request features incorrect data types, missing fields, or unexpected enum categories.

**Solution**: We defined a rigid `CustomerData` schema using Pydantic's `BaseModel` and `Field` validators.
- **Pre-validation**: Before the payload data ever touches the Pandas DataFrame or the Scikit-Learn pipeline, Pydantic ensures the types match our strict specifications (e.g., `SeniorCitizen` must be exactly `0` or `1`, `MonthlyCharges` must be a float).
- **Self-Documenting API**: The API automatically generates an OpenAPI/Swagger UI at `/docs` mapping strict schema constraints, making it exceptionally easy for frontend engineers to integrate our ML service reliably.

---

## 4.  Dockerization Strategy for Deployment (`Dockerfile`)
**Problem**: "It works on my machine" is the nemesis of MLOps. An ML API relies on exact versions of system libraries (like NumPy, scikit-learn, and Pandas) and OS-level dependencies.

**Solution**: The FastAPI service is completely encapsulated within a robust Docker container.
- **Minimal Base Image**: We use the lightweight `python:3.9-slim` to reduce container bloat, lower deployment times, and keep the security vulnerability footprint small.
- **Optimized Layer Caching**: The `COPY requirements.txt` and `RUN pip install` steps are deliberately placed *before* copying the dynamic application code. This optimizes the Docker build process—if application code changes but requirements don't, Docker relies heavily on the cached dependency layer, drastically speeding up CI/CD build times.
