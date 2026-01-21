variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "dooh-ml"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "location" {
  description = "Azure region for resources"
  type        = string
  default     = "eastus"
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}

# Database variables
variable "database_sku" {
  description = "SKU for PostgreSQL Flexible Server"
  type        = string
  default     = "B_Standard_B1ms" # Burstable, cost-effective for dev
}

variable "database_storage_mb" {
  description = "Storage size for PostgreSQL in MB"
  type        = number
  default     = 32768 # 32 GB
}

variable "database_admin_username" {
  description = "Admin username for PostgreSQL"
  type        = string
  default     = "pgadmin"
}

# ML Workspace variables
variable "ml_workspace_sku" {
  description = "SKU for Azure ML Workspace"
  type        = string
  default     = "Basic"
}

# Container Apps variables
variable "container_apps_environment_sku" {
  description = "SKU for Container Apps Environment"
  type        = string
  default     = "Consumption" # Pay-per-use, good for dev
}
