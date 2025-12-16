# Run Instructions (Docker)

All commands should be run from the root directory of the project.

## 1. Prerequisites

Ensure the Docker stack is running:
```cmd
docker-compose up -d
```

## 2. Run Evaluation Script

The evaluation script is run inside the `api` service container.

### Basic Usage
To run a full evaluation (retrieval and answer quality):
```cmd
docker-compose exec api python scripts/run_evaluation.py
```

### Common Options

**Run only retrieval evaluation:**
```cmd
docker-compose exec api python scripts/run_evaluation.py --retrieval-only
```

**Run only answer quality evaluation:**
```cmd
docker-compose exec api python scripts/run_evaluation.py --answer-only
```

**Save results as a baseline:**
```cmd
docker-compose exec api python scripts/run_evaluation.py --baseline
```

**Compare against a previous baseline:**
```cmd
docker-compose exec api python scripts/run_evaluation.py --compare-to data/eval/example_baseline.json
```

**Run a specific test case (by ID):**
```cmd
docker-compose exec api python scripts/run_evaluation.py --test-id answer-001
```

---

## 3. Run Pytest Tests

Tests are run using the dedicated `test` service. These commands spin up a temporary test container.

### Run Unit Tests (Default)
This runs tests excluding those marked as `integration`.
```cmd
docker-compose run --rm test
```

### Run All Tests (Including Integration)
```cmd
docker-compose run --rm test pytest
```

### Run Specific Test File
```cmd
docker-compose run --rm test pytest tests/test_retrieval.py
```

### Run Specific Test Function
```cmd
docker-compose run --rm test pytest tests/test_retrieval.py::TestSemanticSearch::test_search_returns_chunks
```

### Run with Output (Verbose)
```cmd
docker-compose run --rm test pytest -v -s
```

## 4. Troubleshooting

If you encounter "service not found" errors, make sure you are in the correct directory and the services are up:
```cmd
docker-compose ps
```

To view logs for the API service:
```cmd
docker-compose logs -f api
```
