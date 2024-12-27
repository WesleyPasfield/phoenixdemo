# LLM Evaluation System

A system for evaluating LLM outputs using manual reviews and automated evaluations with single or multiple models using Arize Phoenix

## Features

1. Manual content review with customizable metrics
2. Single-model automated evaluations
3. Multi-model consensus evaluations
4. Support for various evaluation types:

    - Toxicity
    - Hallucination
    - Internal Policy Compliance
    - Functional Accuracy

## Prerequisites

1. Docker
2. AWS Account with configured credentials
3. Python 3.11+

## Installation

1. Clone the repository:
```bash
git clone 
cd llm-evaluation
```

2. Deploy AWS resources:
```bash
cd infrastructure
./deploy.sh
```

3. Build Docker image:
```bash
docker build -t llm-eval .
```

## Configuration

- Note the AWS portion of this hasn't been tested via docker, it works from the CLI locally

## Usage

### Running Manual Evaluations

```python
from evaluation_system import LLMEvaluationSystem, ScoreType

# Initialize system
eval_system = LLMEvaluationSystem(
    stack_name="llm-eval",
    region="us-west-2"
)

# Store manual review
eval_system.store_manual_review(
    score_type=ScoreType.FUNCTIONAL_ACCURACY,
    response_id="resp_1",
    project_id="test_project",
    reviewer_id="human_1",
    user_input="What is 2+2?",
    model_response="The sum of 2 and 2 is 4",
    score=1.0,
    explanation="Response is mathematically correct and clear"
)
```

### Single Model Evaluation

```python
import pandas as pd
from evaluation_system import LLMEvaluationSystem, ScoreType

eval_system = LLMEvaluationSystem(
    stack_name="llm-eval",
    region="us-west-2"
)

# Test data
responses_df = pd.DataFrame({
    'response_id': ['resp_1'],
    'user_input': ['What is 2+2?'],
    'response': ['The sum of 2 and 2 is 4']
})

# Run evaluation
results = eval_system.evaluate(
    score_type=ScoreType.FUNCTIONAL_ACCURACY,
    model=eval_system.openai_model,
    responses=responses_df,
    project_id='test_project'
)
```

### Multi-Model Consensus Evaluation

```python
# Run evaluation with multiple models
consensus_results = eval_system.evaluate_multi_model(
    score_type=ScoreType.FUNCTIONAL_ACCURACY,
    models=eval_system.models,  # Uses both OpenAI and Anthropic
    responses=responses_df,
    project_id='test_project',
    consensus_threshold=0.7
)
```

## Output Schema

Evaluations are stored in S3 with the following schema:
```json
{
    "response_id": "string",
    "project_id": "string",
    "reviewer_id": "string",
    "user_input": "string",
    "model_response": "string",
    "correction": "string (optional)",
    "explanation": "string (optional)",
    "score": "float",
    "score_type": "enum (toxicity, hallucination, internal_policy, functional_accuracy)",
    "review_type": "string (manual or llm)",
    "review_model": "string (human or model provider)",
    "timestamp": "ISO-8601 timestamp"
}
```

## Docker Usage

Run with AWS credentials:
```bash
docker run \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -e AWS_SESSION_TOKEN=$AWS_SESSION_TOKEN \
  llm-eval
```

## Directory Structure

```
.
├── Dockerfile
├── infrastructure/
│   ├── deploy.sh
│   ├── template.yaml
│   └── update.sh
├── requirements.txt
└── src/
    ├── evaluation_system.py
    └── run_evaluation.py
```


## Manual Steps

- Set up OpenAI & Anthropic credentials in AWS Secrets
- _get_api_keys() might have some strangeness as I needed to make a slight change to the secretstring returned
- I'm launching an S3 bucket that is referenced throughout - that could be hard coded if not able to use cloudformation in this capacity
- This process uses 4 metrics - any additions or changes would require some manual updates, notably around the parsing of the output to ensure consistency of scores:
    1. Hallucinations
    2. Toxicity
    3. Internal Policy
    4. Functional accuracy
- Project ID & Response ID are hand-waved over right now - making sure they are logical will be important

## Definite TO DOs

1. Enable quering of S3 data from Athena to evaluate outputs
2. Add more input metadata (input model etc...)
3. More comprehensive way of handling project/response/user IDs
4. Unit Tests

## Optional TO DOs

1. Incorporate database storage in addition to S3
2. Make any other needed changes to the schema
3. Investigate why Claude is returning `NOT_PARSABLE` for `llm_classify` requests (explanations seem fine)
4. Enable docker to work w/ AWS creds in secure fashion
