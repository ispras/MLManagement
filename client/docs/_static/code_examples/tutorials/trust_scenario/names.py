import os


eval_job_name = os.environ.get("TRUST_JOB_NAME_EVAL", "trust-scenario-eval-docs-test")
attack_job_name = os.environ.get("TRUST_JOB_NAME_ATTACK", "trust-scenario-attack-docs-test")
model_name = os.environ.get("TRUST_MODEL_NAME", "Pytorch_resnet18_model")
dataset_loader_name = os.environ.get("TRUST_DATASETLOADER_NAME", "CIFAR10")
bucket_name = os.environ.get("TRUST_BUCKET_NAME", "cifar-10")
attacked_data = os.environ.get("TRUST_ATTACKED_BUCKET_NAME", "poisoned-data-docs")
attacked_examples = os.environ.get("TRUST_EXAMPLES_BUCKET_NAME", "poisoned-example-docs")