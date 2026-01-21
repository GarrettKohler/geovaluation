# Container Apps Module
# Creates: Container Apps Environment for scoring APIs

# Container Apps Environment
resource "azurerm_container_app_environment" "main" {
  name                       = "cae-${var.name_prefix}"
  resource_group_name        = var.resource_group_name
  location                   = var.location
  log_analytics_workspace_id = var.log_analytics_workspace_id

  tags = var.tags
}

# Example scoring API container app (commented out until image is ready)
# Uncomment and customize when you have a scoring API image
#
# resource "azurerm_container_app" "scoring_api" {
#   name                         = "ca-scoring-api"
#   container_app_environment_id = azurerm_container_app_environment.main.id
#   resource_group_name          = var.resource_group_name
#   revision_mode                = "Single"
#
#   template {
#     container {
#       name   = "scoring-api"
#       image  = "mcr.microsoft.com/azuredocs/containerapps-helloworld:latest"  # Placeholder
#       cpu    = 0.5
#       memory = "1Gi"
#
#       env {
#         name  = "PORT"
#         value = "8000"
#       }
#     }
#
#     min_replicas = 0
#     max_replicas = 3
#   }
#
#   ingress {
#     external_enabled = true
#     target_port      = 8000
#     traffic_weight {
#       percentage      = 100
#       latest_revision = true
#     }
#   }
#
#   tags = var.tags
# }
