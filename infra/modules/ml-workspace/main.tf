# ML Workspace Module
# Creates: Azure ML Workspace, Application Insights, Log Analytics

# Log Analytics Workspace
resource "azurerm_log_analytics_workspace" "main" {
  name                = "log-${var.name_prefix}-${var.random_suffix}"
  resource_group_name = var.resource_group_name
  location            = var.location
  sku                 = "PerGB2018"
  retention_in_days   = 30

  tags = var.tags
}

# Application Insights for ML Workspace
resource "azurerm_application_insights" "main" {
  name                = "appi-${var.name_prefix}-${var.random_suffix}"
  resource_group_name = var.resource_group_name
  location            = var.location
  workspace_id        = azurerm_log_analytics_workspace.main.id
  application_type    = "other"

  tags = var.tags
}

# Azure Machine Learning Workspace
resource "azurerm_machine_learning_workspace" "main" {
  name                    = "mlw-${var.name_prefix}-${var.random_suffix}"
  resource_group_name     = var.resource_group_name
  location                = var.location
  application_insights_id = azurerm_application_insights.main.id
  key_vault_id            = var.key_vault_id
  storage_account_id      = var.storage_account_id
  container_registry_id   = var.container_registry_id

  sku_name = var.sku_name

  identity {
    type = "SystemAssigned"
  }

  tags = var.tags
}

# Compute instance for development (optional, commented out to save costs)
# Uncomment when needed for interactive development
#
# resource "azurerm_machine_learning_compute_instance" "dev" {
#   name                          = "ci-dev-${var.random_suffix}"
#   machine_learning_workspace_id = azurerm_machine_learning_workspace.main.id
#   virtual_machine_size          = "Standard_DS2_v2"
#
#   authorization_type = "personal"
#
#   tags = var.tags
# }

# Compute cluster for training (optional, commented out to save costs)
# Uncomment when ready for model training
#
# resource "azurerm_machine_learning_compute_cluster" "training" {
#   name                          = "cc-training"
#   machine_learning_workspace_id = azurerm_machine_learning_workspace.main.id
#   location                      = var.location
#   vm_priority                   = "LowPriority"  # Cost savings
#   vm_size                       = "Standard_DS3_v2"
#
#   scale_settings {
#     min_node_count                       = 0
#     max_node_count                       = 4
#     scale_down_nodes_after_idle_duration = "PT5M"  # 5 minutes
#   }
#
#   identity {
#     type = "SystemAssigned"
#   }
#
#   tags = var.tags
# }
