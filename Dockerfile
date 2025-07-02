FROM nvcr.io/nvidia/tritonserver:25.04-py3

# Install SageMaker requirements
RUN pip install sagemaker-inference tritonclient[http] opencv-python-headless

# Create model directory (SageMaker will mount models here)
RUN mkdir -p /opt/ml/model

#COPY model_repository /opt/ml/model

# Copy your custom serve script
COPY serve /opt/serve
RUN chmod +x /opt/serve
EXPOSE 8080
# Set proper entrypoint
ENTRYPOINT ["/opt/serve"]