# DOOH ML Platform Infrastructure

Terraform configuration for deploying the DOOH Site Optimization ML platform on Azure.

## Architecture

```
Resource Group (rg-dooh-ml-dev)
├── Storage Account (Data Lake Gen2)
│   ├── raw-data/          # Source site data
│   ├── processed-data/    # Transformed datasets
│   ├── models/            # Model artifacts
│   └── mlflow/            # MLflow tracking
├── Key Vault              # Secrets management
├── Container Registry     # Docker images
├── PostgreSQL + PostGIS   # Geospatial database
├── Azure ML Workspace     # Model training & registry
│   ├── Application Insights
│   └── Log Analytics
└── Container Apps Environment  # Scoring APIs
```

## Prerequisites

1. [Terraform](https://terraform.io/downloads) >= 1.5.0
2. [Azure CLI](https://docs.microsoft.com/cli/azure/install-azure-cli)
3. Azure subscription with Owner/Contributor access

## Quick Start

```bash
# 1. Login to Azure
az login
az account set --subscription "Your-Subscription"

# 2. Initialize Terraform
cd infra
terraform init

# 3. Review plan
terraform plan

# 4. Deploy
terraform apply
```

## Configuration

Copy and customize the example variables file:

```bash
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values
```

### Key Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `project_name` | `dooh-ml` | Prefix for all resources |
| `environment` | `dev` | Environment (dev/staging/prod) |
| `location` | `eastus` | Azure region |
| `database_sku` | `B_Standard_B1ms` | PostgreSQL SKU |

## Modules

- **storage/** - Storage Account, Key Vault, Container Registry
- **database/** - PostgreSQL Flexible Server with PostGIS
- **ml-workspace/** - Azure ML Workspace, Log Analytics, App Insights
- **container-apps/** - Container Apps Environment for APIs

## Remote State (Recommended)

For team collaboration, configure remote state:

1. Create a storage account for state
2. Copy `backend.tf.example` to `backend.tf`
3. Update with your storage account details
4. Run `terraform init -migrate-state`

## Outputs

After deployment, get connection info:

```bash
# PostgreSQL connection
terraform output postgresql_server_fqdn

# ML Workspace
terraform output ml_workspace_name

# Container Registry
terraform output container_registry_login_server
```

## Cost Optimization

Default configuration uses cost-effective SKUs for development:

- PostgreSQL: Burstable B1ms (~$12/month)
- Container Registry: Basic (~$5/month)
- Container Apps: Consumption (pay-per-use)
- ML Workspace: Basic (no additional cost)

For production, update SKUs in `terraform.tfvars`.

## Cleanup

```bash
terraform destroy
```
