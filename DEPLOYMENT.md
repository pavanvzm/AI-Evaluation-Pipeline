# AI Evaluation Pipeline - Deployment Guide

This guide covers deploying the AI Evaluation Pipeline to production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Docker Deployment](#docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Configuration](#configuration)
5. [Monitoring](#monitoring)
6. [Security](#security)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Tools

- Docker 20.10+
- Docker Compose 2.0+ (or `docker compose`)
- Kubernetes 1.25+ (for K8s deployment)
-kubectl 1.25+ (for K8s deployment)

### Required Accounts

- OpenAI API key
- Anthropic API key
- Groq API key

---

## Docker Deployment

### Quick Start (Development)

```bash
# Clone and configure
git clone <repository-url>
cd AI-Evaluation-Pipeline

# Copy environment file
cp .env.example .env
# Edit .env with your API keys

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

### Production Deployment

```bash
# Use production environment
cp .env.production .env
# Edit .env with secure passwords and API keys

# Build optimized image
docker build -t ai-evaluation-pipeline:latest --target production .

# Start with production configuration
docker-compose -f docker-compose.yml up -d

# Scale API workers (requires restart)
docker-compose up -d --scale api=3
```

### Service URLs

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

### Stopping Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (full cleanup)
docker-compose down -v
```

---

## Kubernetes Deployment

### Prerequisites

1. Kubernetes cluster (GKE, EKS, AKS, or self-hosted)
2. `kubectl` configured with cluster access
3. Helm 3+ (optional, for Helm deployments)

### Step 1: Prepare Secrets

```bash
# Create namespace
kubectl apply -f deploy/kubernetes/00-namespace.yaml

# Update secrets with real values
cat <<EOF > /tmp/secrets-patch.yaml
stringData:
  OPENAI_API_KEY: "sk-your-key"
  ANTHROPIC_API_KEY: "sk-ant-your-key"
  GROQ_API_KEY: "gsk-your-key"
  POSTGRES_PASSWORD: "secure-password-here"
EOF

# Apply secrets
kubectl apply -f deploy/kubernetes/00-namespace.yaml
kubectl patch secret ai-eval-secrets --namespace ai-evaluation --patch "$(cat /tmp/secrets-patch.yaml)"
```

### Step 2: Update Configuration

Edit `deploy/kubernetes/01-deployment.yaml` to update:
- Storage class for your cloud provider
- Database connection string
- Resource limits

### Step 3: Deploy

```bash
# Deploy namespace and config
kubectl apply -f deploy/kubernetes/00-namespace.yaml

# Deploy PostgreSQL
kubectl apply -f deploy/kubernetes/02-postgres.yaml

# Wait for PostgreSQL to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n ai-evaluation --timeout=120s

# Deploy API and Dashboard
kubectl apply -f deploy/kubernetes/01-deployment.yaml

# Check deployment status
kubectl get pods -n ai-evaluation
```

### Step 4: Verify

```bash
# Port forward for local testing
kubectl port-forward -n ai-evaluation svc/ai-eval-api 8000:8000

# Test health endpoint
curl http://localhost:8000/health

# Get all services
kubectl get svc -n ai-evaluation
```

### Scaling

```bash
# Scale API
kubectl scale deployment ai-eval-api --replicas=5 -n ai-evaluation

# Enable HPA (auto-scaling)
kubectl autoscale deployment ai-eval-api --min=2 --max=10 --cpu-percent=70 -n ai-evaluation
```

### Ingress Configuration

Update the ingress manifests with your domain:

```bash
# Edit ingress to use your domain
kubectl edit ingress ai-eval-api-ingress -n ai-evaluation
# Change api.your-domain.com to your actual domain
```

---

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | Yes |
| `ANTHROPIC_API_KEY` | Anthropic API key | Yes |
| `GROQ_API_KEY` | Groq API key | Yes |
| `POSTGRES_USER` | PostgreSQL username | No (default: ai_eval) |
| `POSTGRES_PASSWORD` | PostgreSQL password | Yes |
| `POSTGRES_DB` | Database name | No (default: evaluation) |
| `LOG_LEVEL` | Logging level | No (default: INFO) |
| `MAX_CONCURRENT_REQUESTS` | Max parallel API calls | No (default: 10) |
| `REQUEST_TIMEOUT` | API timeout (seconds) | No (default: 120) |

### Database Connection

For SQLite (development):
```bash
DATABASE_URL=sqlite:///data/evaluation_results.db
```

For PostgreSQL (production):
```bash
DATABASE_URL=postgresql://user:password@host:5432/evaluation
```

### Model Pricing

Update `config/config.yaml` to match current provider pricing:

```yaml
models:
  openai:
    models:
      - name: "gpt-4o"
        pricing:
          input_cost_per_1k: 0.005  # Per 1K input tokens
          output_cost_per_1k: 0.015  # Per 1K output tokens
```

---

## Monitoring

### Prometheus Metrics

Access Prometheus at `http://localhost:9090` (or your K8s ingress).

Key metrics:
- `ai_eval_requests_total` - Total API requests
- `ai_eval_request_duration_seconds` - Request latency
- `ai_eval_evaluations_total` - Completed evaluations
- `ai_eval_active_runs` - Currently running evaluations

### Grafana Dashboards

1. Access Grafana at `http://localhost:3000`
2. Login with `admin` / password from `.env`
3. Navigate to Dashboards > AI Evaluation

Pre-configured dashboards:
- **API Overview** - Request rates, latency, errors
- **Evaluation Metrics** - Accuracy, faithfulness, costs
- **Resource Usage** - CPU, memory, network

### Alerting

Set up alerts in Grafana:

```yaml
# Example alert rule
- alert: HighErrorRate
  expr: rate(ai_eval_requests_total{status="500"}[5m]) > 0.1
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "High error rate on AI Evaluation API"
```

---

## Security

### API Security

1. **Use HTTPS**: Always deploy behind TLS termination
2. **API Keys**: Store in Kubernetes secrets, never in code
3. **Rate Limiting**: Configured in nginx (100 req/s by default)
4. **CORS**: Configure allowed origins in production

### Database Security

1. **Strong Passwords**: Use 32+ character random passwords
2. **Network Isolation**: Use internal Kubernetes networking
3. **Encryption**: Enable SSL for database connections
4. **Backups**: Configure regular automated backups

### Container Security

1. **Non-root User**: Containers run as non-root (uid 1000)
2. **Read-only Filesystem**: Consider enabling read-only root
3. **Image Scanning**: Scan images for vulnerabilities
4. **Minimal Base**: Use distroless/scratch images when possible

---

## Troubleshooting

### Container Issues

```bash
# Check container logs
docker-compose logs -f api

# Check container status
docker-compose ps

# Restart a service
docker-compose restart api

# Rebuild and restart
docker-compose up -d --build api
```

### Kubernetes Issues

```bash
# Check pod status
kubectl get pods -n ai-evaluation

# View pod logs
kubectl logs -n ai-evaluation deployment/ai-eval-api

# Describe pod for events
kubectl describe pod -n ai-evaluation <pod-name>

# Check resource usage
kubectl top pods -n ai-evaluation
```

### Database Issues

```bash
# Connect to PostgreSQL
kubectl exec -it -n ai-evaluation postgres-0 -- psql -U ai_eval

# Check database size
SELECT pg_size_pretty(pg_database_size('evaluation'));

# View active connections
SELECT * FROM pg_stat_activity;
```

### Performance Issues

1. **High Latency**: Check API worker count and resource limits
2. **OOM Errors**: Increase memory limits
3. **Slow Queries**: Enable query logging, add indexes
4. **Rate Limiting**: Consider increasing limits or scaling

---

## Backup & Recovery

### Database Backup

```bash
# Local backup
docker-compose exec postgres pg_dump -U ai_eval evaluation > backup.sql

# Kubernetes backup
kubectl exec -it -n ai-evaluation postgres-0 -- pg_dump -U ai_eval evaluation > backup.sql
```

### Database Restore

```bash
# Local restore
docker-compose exec -T postgres psql -U ai_eval evaluation < backup.sql

# Kubernetes restore
kubectl exec -i -n ai-evaluation postgres-0 -- psql -U ai_eval evaluation < backup.sql
```

### Disaster Recovery

1. Stop all services
2. Restore database from backup
3. Verify data integrity
4. Restart services

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Configure kubectl
        run: |
          echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > kubeconfig
          echo "KUBECONFIG=$(pwd)/kubeconfig" >> $GITHUB_ENV
      
      - name: Deploy to Kubernetes
        run: |
          kubectl apply -f deploy/kubernetes/
          kubectl rollout status deployment/ai-eval-api -n ai-evaluation
```

---

## Support

For issues or questions:
- GitHub Issues: https://github.com/your-org/ai-evaluation-pipeline/issues
- Documentation: https://docs.your-domain.com