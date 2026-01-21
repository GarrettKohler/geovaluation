output "resource_group_name" {
  description = "Name of the resource group"
  value       = azurerm_resource_group.main.name
}

output "resource_group_id" {
  description = "ID of the resource group"
  value       = azurerm_resource_group.main.id
}

# Storage outputs
output "storage_account_name" {
  description = "Name of the storage account"
  value       = module.storage.storage_account_name
}

output "storage_account_id" {
  description = "ID of the storage account"
  value       = module.storage.storage_account_id
}

output "key_vault_name" {
  description = "Name of the Key Vault"
  value       = module.storage.key_vault_name
}

output "container_registry_name" {
  description = "Name of the Container Registry"
  value       = module.storage.container_registry_name
}

output "container_registry_login_server" {
  description = "Login server for Container Registry"
  value       = module.storage.container_registry_login_server
}

# Database outputs
output "postgresql_server_name" {
  description = "Name of the PostgreSQL server"
  value       = module.database.server_name
}

output "postgresql_server_fqdn" {
  description = "FQDN of the PostgreSQL server"
  value       = module.database.server_fqdn
}

output "postgresql_database_name" {
  description = "Name of the PostgreSQL database"
  value       = module.database.database_name
}

# ML Workspace outputs
output "ml_workspace_name" {
  description = "Name of the Azure ML Workspace"
  value       = module.ml_workspace.workspace_name
}

output "ml_workspace_id" {
  description = "ID of the Azure ML Workspace"
  value       = module.ml_workspace.workspace_id
}

# Container Apps outputs
output "container_apps_environment_name" {
  description = "Name of the Container Apps Environment"
  value       = module.container_apps.environment_name
}

output "container_apps_environment_id" {
  description = "ID of the Container Apps Environment"
  value       = module.container_apps.environment_id
}
