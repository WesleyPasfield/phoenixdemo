from phoenix import Session, StorageConfig, S3LogConfig
from phoenix.evaluation import EvaluationFramework, ModelBasedEvaluator, ManualEvaluator
from phoenix.store import DocumentStore
from dataclasses import dataclass
from typing import Dict, List
import os
import boto3
from botocore.exceptions import ClientError

@dataclass
class Review:
   response_id: str
   score: int  # 1-5
   correction: str
   explanation: str
   reviewer_id: str

class LLMEvaluationSystem:
   def __init__(self):
       self.secrets = self._get_secrets()
       self.db_config = StorageConfig(
           type="postgres",
           connection_string=f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('RDS_HOST')}/llm_eval"
       )

       self.s3_config = S3LogConfig(
           bucket=os.getenv('S3_BUCKET'),
           prefix="evaluations/",
           region=os.getenv('AWS_REGION')
       )

       self.session = Session(
           project_dir="llm_eval",
           storage=self.db_config
       )
       
       self.session.configure_logging(
           handlers=["s3"],
           s3_config=self.s3_config
       )

       self.store = DocumentStore(encryption=True)
       
       self.evaluators = {
           "gpt-4": ModelBasedEvaluator(
               model="gpt-4",
               api_key=self.secrets["OPENAI_API_KEY"],
               api_base="https://api.openai.com/v1"
           ),
           "claude-3": ModelBasedEvaluator(
               model="claude-3", 
               api_key=self.secrets["ANTHROPIC_API_KEY"],
               api_base="https://api.anthropic.com/v1"
           )
       }
       self.internal_policy_metric = CustomMetric(
           name="internal_policy",
           evaluator="gpt-4",
           prompt_template="Evaluate if this response complies with internal policies: {response}. Score 1-5 where 1=non-compliant, 5=fully compliant. Include specific policy violations if any."
       )
       self.framework = EvaluationFramework(
           evaluators=list(self.evaluators.values()),
           metrics=["relevance", "coherence", "factual_accuracy", self.internal_policy_metric]
       )
       self.pipeline = self.session.create_pipeline(
           store=self.store,
           framework=self.framework,
           auto_persist=True
       )
       self.pipeline.add_output_handler(
           handler_type="s3",
           config={
               "bucket": os.getenv('S3_BUCKET'),
               "prefix": "results/",
               "region": os.getenv('AWS_REGION')
           }
       )
       def _get_secrets(self):
           session = boto3.session.Session()
           client = session.client('secretsmanager')
           try:
               response = client.get_secret_value(
                   SecretId='llm-eval/api-keys'
                   )
               return json.loads(response['SecretString'])
           except ClientError as e:
               print(f"Error accessing secrets: {e}")
               raise



   def add_manual_review(self, review: Review):
       result = {
           "response_id": review.response_id,
           "score": review.score,
           "correction": review.correction,
           "explanation": review.explanation,
           "reviewer_id": review.reviewer_id,
           "type": "manual"
       }
       self.pipeline.store_result(result)
       return result

   def evaluate_with_llms(self, response: str, models: List[str] = None):
       if not models:
           models = list(self.evaluators.keys())
           
       results = {}
       for model in models:
           if model in self.evaluators:
               results[model] = self.evaluators[model].evaluate(response)
       
       self.pipeline.store_result({
           "response": response,
           "llm_evaluations": results,
           "type": "llm"
       })
       return results

   def get_results(self, response_id: str = None):
       return self.session.get_results(response_id)