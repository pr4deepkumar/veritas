#!/usr/bin/env bash
# Build the VERITAS image, push it to ECR, and roll the ECS service onto it.
#
# Usage: ./scripts/deploy_aws.sh
# Optional environment: AWS_REGION, APP_NAME, ENVIRONMENT, PLATFORM

set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
APP_NAME="${APP_NAME:-veritas}"
ENVIRONMENT="${ENVIRONMENT:-prod}"
# Must match the task's cpu_architecture in terraform/ (X86_64 by default). Without it, a build
# on an Apple Silicon Mac produces an arm64 image that Fargate rejects with "exec format error".
PLATFORM="${PLATFORM:-linux/amd64}"

cd "$(dirname "$0")/.."  # the build context is the repository root

echo "=== VERITAS AWS deployment ==="
AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
API_REPO="${ECR_REGISTRY}/${APP_NAME}-api"
IMAGE_TAG="$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)"
echo "Account: ${AWS_ACCOUNT_ID}  Region: ${AWS_REGION}  Platform: ${PLATFORM}  Tag: ${IMAGE_TAG}"

echo "--> Logging in to ECR"
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

echo "--> Building ${API_REPO}:${IMAGE_TAG}"
docker build --platform "${PLATFORM}" -f docker/Dockerfile.api \
  -t "${API_REPO}:${IMAGE_TAG}" -t "${API_REPO}:latest" .

echo "--> Pushing"
docker push "${API_REPO}:${IMAGE_TAG}"
docker push "${API_REPO}:latest"

# The task definition references :latest, so `terraform apply` alone does not pick up a new
# push; force a deployment so ECS pulls the image again.
CLUSTER="${APP_NAME}-cluster-${ENVIRONMENT}"
SERVICE="${APP_NAME}-api-service"
ACTIVE_SERVICE="$(aws ecs describe-services --region "${AWS_REGION}" --cluster "${CLUSTER}" \
  --services "${SERVICE}" --query 'services[?status==`ACTIVE`].serviceName' --output text 2>/dev/null || true)"
if [[ "${ACTIVE_SERVICE}" == "${SERVICE}" ]]; then
  echo "--> Redeploying ${SERVICE}"
  aws ecs update-service --region "${AWS_REGION}" --cluster "${CLUSTER}" --service "${SERVICE}" \
    --force-new-deployment > /dev/null
  echo "Rollout started. Wait for it with:"
  echo "  aws ecs wait services-stable --region ${AWS_REGION} --cluster ${CLUSTER} --services ${SERVICE}"
else
  echo "ECS service ${SERVICE} not found in ${CLUSTER}; run 'terraform apply' in terraform/ to create it."
fi

echo "=== Pushed ${API_REPO}:${IMAGE_TAG} ==="
echo "To pin this build instead of :latest: terraform apply -var image_tag=${IMAGE_TAG}"
