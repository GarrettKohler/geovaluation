# Storage Module
# Creates: Storage Account, Key Vault, Container Registry

data "azurerm_client_config" "current" {}

# Storage Account - Data lake for site data and model artifacts
resource "azurerm_storage_account" "main" {
  name                     = "st${replace(var.name_prefix, "-", "")}${var.random_suffix}"
  resource_group_name      = var.resource_group_name
  location                 = var.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  account_kind             = "StorageV2"
  is_hns_enabled           = true # Enable hierarchical namespace for Data Lake Gen2

  blob_properties {
    versioning_enabled = true
  }

  tags = var.tags
}

# Storage containers for different data types
resource "azurerm_storage_container" "raw_data" {
  name                  = "raw-data"
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "processed_data" {
  name                  = "processed-data"
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "models" {
  name                  = "models"
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "mlflow" {
  name                  = "mlflow"
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private"
}

# Key Vault - Secrets management
resource "azurerm_key_vault" "main" {
  name                       = "kv-${var.name_prefix}-${var.random_suffix}"
  resource_group_name        = var.resource_group_name
  location                   = var.location
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = "standard"
  soft_delete_retention_days = 7
  purge_protection_enabled   = false # Set to true for prod

  # Allow current user to manage secrets
  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azurerm_client_config.current.object_id

    secret_permissions = [
      "Get", "List", "Set", "Delete", "Purge", "Recover"
    ]

    key_permissions = [
      "Get", "List", "Create", "Delete", "Purge", "Recover"
    ]
  }

  tags = var.tags
}

# Container Registry - Docker images for ML models and APIs
resource "azurerm_container_registry" "main" {
  name                = "cr${replace(var.name_prefix, "-", "")}${var.random_suffix}"
  resource_group_name = var.resource_group_name
  location            = var.location
  sku                 = "Basic" # Use Standard or Premium for prod
  admin_enabled       = true    # Enable for initial setup, disable for prod

  tags = var.tags
}
