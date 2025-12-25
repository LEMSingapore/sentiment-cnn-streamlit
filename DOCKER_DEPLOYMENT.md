# Docker Deployment Guide

This guide covers deploying the Sentiment CNN Streamlit app using Docker, both locally and on AWS.

## Prerequisites

- Docker installed on your machine
- Docker Compose (optional, for easier local testing)
- AWS account (for AWS deployment)
- AWS CLI configured (for AWS deployment)

## Local Docker Deployment

### Option 1: Using Docker Compose (Recommended for local testing)

```bash
# Build and run the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

The app will be available at `http://localhost:8501`

### Option 2: Using Docker directly

```bash
# Build the image
docker build -t sentiment-cnn-app .

# Run the container
docker run -d -p 8501:8501 --name sentiment-cnn-app sentiment-cnn-app

# View logs
docker logs -f sentiment-cnn-app

# Stop and remove the container
docker stop sentiment-cnn-app
docker rm sentiment-cnn-app
```

## AWS Deployment Options

### Option 1: AWS ECS (Elastic Container Service) with Fargate

1. **Push image to Amazon ECR:**

```bash
# Create ECR repository
aws ecr create-repository --repository-name sentiment-cnn-app --region us-east-1

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag your image
docker tag sentiment-cnn-app:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/sentiment-cnn-app:latest

# Push to ECR
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/sentiment-cnn-app:latest
```

2. **Create ECS Task Definition** (save as `task-definition.json`):

See the `aws-ecs/task-definition.json` file for the complete configuration.

3. **Deploy to ECS:**

```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name sentiment-cnn-cluster --region us-east-1

# Register task definition
aws ecs register-task-definition --cli-input-json file://aws-ecs/task-definition.json

# Create service
aws ecs create-service \
  --cluster sentiment-cnn-cluster \
  --service-name sentiment-cnn-service \
  --task-definition sentiment-cnn-task \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

### Option 2: AWS App Runner (Simpler, fully managed)

```bash
# Create App Runner service
aws apprunner create-service \
  --service-name sentiment-cnn-app \
  --source-configuration "{\"ImageRepository\":{\"ImageIdentifier\":\"<account-id>.dkr.ecr.us-east-1.amazonaws.com/sentiment-cnn-app:latest\",\"ImageRepositoryType\":\"ECR\",\"ImageConfiguration\":{\"Port\":\"8501\"}}}" \
  --instance-configuration "Cpu=1024,Memory=2048" \
  --region us-east-1
```

### Option 3: AWS Lightsail Containers (Cost-effective for small apps)

```bash
# Create container service
aws lightsail create-container-service \
  --service-name sentiment-cnn-app \
  --power small \
  --scale 1 \
  --region us-east-1

# Push and deploy
aws lightsail push-container-image \
  --service-name sentiment-cnn-app \
  --label sentiment-cnn-app \
  --image sentiment-cnn-app:latest

# Create deployment
aws lightsail create-container-service-deployment \
  --service-name sentiment-cnn-app \
  --containers file://aws-lightsail/containers.json \
  --public-endpoint file://aws-lightsail/public-endpoint.json
```

## Environment Variables

You can configure the following environment variables:

- `STREAMLIT_SERVER_PORT` - Port for Streamlit (default: 8501)
- `STREAMLIT_SERVER_ADDRESS` - Server address (default: 0.0.0.0)
- `STREAMLIT_SERVER_HEADLESS` - Run in headless mode (default: true)

## Troubleshooting

### Container fails to start
- Check logs: `docker logs sentiment-cnn-app`
- Verify all model files are present
- Ensure sufficient memory (min 2GB recommended)

### NLTK data not found
- NLTK data is downloaded during image build
- If issues persist, rebuild the image: `docker-compose build --no-cache`

### Model file too large for Git LFS
- The Docker image includes the model file directly
- Ensure Git LFS is properly configured when cloning the repo

## Cost Optimization on AWS

- **Development**: Use Lightsail Containers ($7-40/month for small instances)
- **Production**: Use ECS Fargate with auto-scaling based on load
- **Low Traffic**: Use App Runner with automatic pause/resume

## Security Considerations

- Use AWS Secrets Manager for sensitive configurations
- Enable VPC endpoints for private ECR access
- Use IAM roles instead of access keys
- Enable CloudWatch logging for monitoring
