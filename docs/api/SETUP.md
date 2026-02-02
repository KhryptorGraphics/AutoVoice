# API Documentation Setup Guide

Quick setup guide for AutoVoice API documentation.

## Installation

### 1. Install Documentation Dependencies

```bash
pip install -r requirements-docs.txt
```

This installs:
- `apispec` - OpenAPI spec generation
- `apispec-webframeworks` - Flask integration
- `flask-swagger-ui` - Swagger UI
- `marshmallow` - Schema validation

### 2. Verify Installation

```bash
python -c "import apispec; import flask_swagger_ui; print('✅ Dependencies installed')"
```

## Running the Documentation

### Start AutoVoice Server

```bash
# Standard startup
python main.py --host 0.0.0.0 --port 5000

# Or with config
python main.py --config config/gpu_config.yaml
```

### Access Documentation

Once the server is running, access:

**Swagger UI (Interactive):**
```
http://localhost:5000/docs
```

**OpenAPI Spec (JSON):**
```
http://localhost:5000/api/v1/openapi.json
```

**OpenAPI Spec (YAML):**
```
http://localhost:5000/api/v1/openapi.yaml
```

## Validate Documentation

Run the validation script:

```bash
# Make script executable
chmod +x scripts/validate_openapi.py

# Run validation
python scripts/validate_openapi.py
```

Expected output:
```
============================================================
AutoVoice OpenAPI Validation
============================================================
✅ Server running at http://localhost:5000

============================================================
Running: OpenAPI JSON Spec
============================================================
Fetching OpenAPI spec from http://localhost:5000/api/v1/openapi.json...
✅ OpenAPI version: 3.0.2
✅ Title: AutoVoice API
✅ Version: 1.0.0
✅ Total endpoints documented: 60
✅ Total schemas defined: 12

✅ Endpoint groups:
   - Audio Processing
   - Configuration
   - Conversion
   - System
   - Training
   - Voice Profiles
   - YouTube

============================================================
Running: OpenAPI YAML Spec
============================================================
Validating YAML spec at http://localhost:5000/api/v1/openapi.yaml...
✅ YAML spec valid and parseable

============================================================
Running: Swagger UI
============================================================
Testing Swagger UI at http://localhost:5000/docs...
✅ Swagger UI accessible at http://localhost:5000/docs

============================================================
Running: Endpoint Coverage
============================================================

✅ Expected endpoint groups:
   - Conversion
   - Voice Profiles
   - Training
   - Audio Processing
   - System
   - YouTube

============================================================
VALIDATION SUMMARY
============================================================
✅ PASS: OpenAPI JSON Spec
✅ PASS: OpenAPI YAML Spec
✅ PASS: Swagger UI
✅ PASS: Endpoint Coverage

🎉 All validation tests passed!

📚 View API documentation at: http://localhost:5000/docs
```

## Import Postman Collection

### Option 1: Postman Desktop

1. Open Postman
2. Click **Import** button
3. Select `docs/api/postman_collection.json`
4. Collection appears in sidebar with all endpoints ready to use

### Option 2: Insomnia

1. Open Insomnia
2. Click **Create** → **Import From** → **File**
3. Select `docs/api/postman_collection.json`
4. Collection imported with all requests

### Configure Variables

Set these collection variables:
- `base_url`: `http://localhost:5000`
- `api_version`: `v1`
- `profile_id`: Your test profile ID
- `job_id`: Will be auto-populated from responses

## Testing the API

### 1. Health Check

```bash
curl http://localhost:5000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "components": {
    "singing_pipeline": "available",
    "voice_cloner": "available",
    "gpu": "available"
  }
}
```

### 2. List Voice Profiles

```bash
curl http://localhost:5000/api/v1/voice/profiles
```

### 3. Create Voice Profile

```bash
curl -X POST http://localhost:5000/api/v1/voice/clone \
  -F "name=Test Profile" \
  -F "samples=@sample1.wav" \
  -F "samples=@sample2.wav" \
  -F "samples=@sample3.wav"
```

### 4. Convert Song

```bash
curl -X POST http://localhost:5000/api/v1/convert/song \
  -F "song=@test_song.mp3" \
  -F "profile_id=YOUR_PROFILE_ID" \
  -F "output_quality=balanced"
```

## Troubleshooting

### Swagger UI Not Loading

**Problem:** `/docs` returns 404

**Solution:**
1. Verify `flask-swagger-ui` is installed: `pip list | grep swagger`
2. Check app.py has registered blueprints:
   ```python
   from .api_docs import docs_bp, swagger_ui_blueprint
   app.register_blueprint(docs_bp, url_prefix='/api/v1')
   app.register_blueprint(swagger_ui_blueprint)
   ```
3. Restart server

### OpenAPI Spec Returns Empty

**Problem:** `/api/v1/openapi.json` returns `{}`

**Solution:**
1. Check for import errors: `python -c "from auto_voice.web.openapi_spec import create_openapi_spec"`
2. Verify apispec is installed: `pip install apispec apispec-webframeworks`
3. Check logs for errors during spec generation

### Validation Script Fails

**Problem:** `validate_openapi.py` reports failures

**Solution:**
1. Ensure server is running: `curl http://localhost:5000/api/v1/health`
2. Install requests: `pip install requests pyyaml`
3. Check firewall isn't blocking localhost:5000

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'flask_swagger_ui'`

**Solution:**
```bash
pip install flask-swagger-ui==4.11.1
```

## Development Workflow

### Updating Documentation

1. **Update Schemas** - Modify `openapi_spec.py`:
   ```python
   class NewFeatureSchema(Schema):
       field = fields.Str(required=True)

   spec.components.schema("NewFeature", schema=NewFeatureSchema)
   ```

2. **Add Endpoint** - Update `api_docs.py`:
   ```python
   spec.path(
       path="/api/v1/new/endpoint",
       operations={
           "post": {
               "tags": ["Feature"],
               "summary": "New endpoint",
               "requestBody": {...},
               "responses": {...}
           }
       }
   )
   ```

3. **Validate Changes**:
   ```bash
   python scripts/validate_openapi.py
   ```

4. **View in Swagger UI**:
   - Refresh http://localhost:5000/docs
   - New endpoint appears in appropriate section

### Testing Endpoint Documentation

1. Open Swagger UI: http://localhost:5000/docs
2. Find your endpoint in the list
3. Click **Try it out**
4. Fill in parameters
5. Click **Execute**
6. Verify response matches documentation

## CI/CD Integration

### GitHub Actions

```yaml
name: Validate API Docs

on: [push, pull_request]

jobs:
  validate-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-docs.txt

      - name: Start server
        run: |
          python main.py --testing &
          sleep 10

      - name: Validate OpenAPI spec
        run: python scripts/validate_openapi.py
```

### Pre-commit Hook

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Validate OpenAPI spec before commit

echo "Validating API documentation..."
python scripts/validate_openapi.py

if [ $? -ne 0 ]; then
    echo "❌ API documentation validation failed"
    echo "Please fix errors before committing"
    exit 1
fi

echo "✅ API documentation valid"
```

Make executable:
```bash
chmod +x .git/hooks/pre-commit
```

## Additional Resources

- [Swagger UI Documentation](https://swagger.io/tools/swagger-ui/)
- [OpenAPI 3.0 Specification](https://swagger.io/specification/)
- [APISpec Documentation](https://apispec.readthedocs.io/)
- [Marshmallow Documentation](https://marshmallow.readthedocs.io/)

## Next Steps

1. ✅ Setup complete - Documentation running at http://localhost:5000/docs
2. 📖 Read [tutorials.md](./tutorials.md) for usage examples
3. 🔌 Review [websocket-events.md](./websocket-events.md) for real-time features
4. 📮 Import [postman_collection.json](./postman_collection.json) for testing
5. 🚀 Start building your integration!
