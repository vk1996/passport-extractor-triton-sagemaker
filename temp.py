import boto3
import os
from sagemaker import get_execution_role


version='1j'
role = get_execution_role()
region=boto3.session.Session().region_name
account_id = boto3.client('sts').get_caller_identity().get('Account')
endpoint_name = "passport-extraction-triton-sagemaker-deployment-"+version
model_bucket_name="passport-extraction-triton-sagemaker-model-bucket"
model_location=f"s3://sagemaker-{region}-{account_id}/{model_bucket_name}/model_repository.tar.gz"
print(model_location)
inference_repository_uri=f"{account_id}.dkr.ecr.{region}.amazonaws.com/{endpoint_name}:test"+version
print(inference_repository_uri)

os.system(f"sudo docker build -t {endpoint_name} .")
os.system(f"sudo docker tag {endpoint_name} {inference_repository_uri}")
os.system(f"aws ecr get-login-password --region {region} | sudo docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com")
os.system(f"aws ecr create-repository --repository-name {endpoint_name}")
os.system(f"sudo docker push {account_id}.dkr.ecr.{region}.amazonaws.com/{endpoint_name}:test{version}")



# import sagemaker
# from sagemaker import get_execution_role
# from sagemaker.model import Model
#
# role = get_execution_role()  # works in SageMaker Notebook or EC2 with proper IAM
#
# sagemaker_session = sagemaker.Session()
#
# # Replace with your values
# ecr_image_uri = inference_repository_uri
# endpoint_name=endpoint_name+"e"
# sagemaker_model = Model(
#             model_data=model_location,
#             role=role,
#             image_uri=inference_repository_uri,
#             name=endpoint_name.lower())
#
# sagemaker_model.deploy(initial_instance_count=1, instance_type='ml.c6i.2xlarge',
#                                    endpoint_name=endpoint_name)
