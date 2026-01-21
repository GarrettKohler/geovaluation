locals {
  name_prefix = "${var.project_name}-${var.environment}"

  common_tags = merge(var.tags, {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
  })
}

# Generate random suffix for globally unique names
resource "random_string" "suffix" {
  length  = 6
  special = false
  upper   = false
}

# Resource Group
resource "azurerm_resource_group" "main" {
  name     = "rg-${local.name_prefix}"
  location = var.location
  tags     = local.common_tags
}

# Storage Module - Data lake and ML artifacts
module "storage" {
  source = "./modules/storage"

  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  name_prefix         = local.name_prefix
  random_suffix       = random_string.suffix.result
  tags                = local.common_tags
}

# Database Module - PostgreSQL with PostGIS
module "database" {
  source = "./modules/database"

  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  name_prefix         = local.name_prefix
  random_suffix       = random_string.suffix.result
  tags                = local.common_tags

  sku_name           = var.database_sku
  storage_mb         = var.database_storage_mb
  admin_username     = var.database_admin_username
  key_vault_id       = module.storage.key_vault_id
}

# ML Workspace Module
module "ml_workspace" {
  source = "./modules/ml-workspace"

  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  name_prefix         = local.name_prefix
  random_suffix       = random_string.suffix.result
  tags                = local.common_tags

  storage_account_id   = module.storage.storage_account_id
  key_vault_id         = module.storage.key_vault_id
  container_registry_id = module.storage.container_registry_id
  sku_name             = var.ml_workspace_sku
}

# Container Apps Module - Scoring API
module "container_apps" {
  source = "./modules/container-apps"

  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  name_prefix         = local.name_prefix
  tags                = local.common_tags

  log_analytics_workspace_id = module.ml_workspace.log_analytics_workspace_id
}

# Database Schema Module - Create tables and PostGIS objects
module "database_schema" {
  source = "./modules/database-schema"

  server_fqdn    = module.database.server_fqdn
  database_name  = module.database.database_name
  admin_username = var.database_admin_username
  admin_password = module.database.admin_password

  depends_on = [module.database]
}
