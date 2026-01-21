output "storage_account_id" {
  description = "ID of the storage account"
  value       = azurerm_storage_account.main.id
}

output "storage_account_name" {
  description = "Name of the storage account"
  value       = azurerm_storage_account.main.name
}

output "storage_account_primary_access_key" {
  description = "Primary access key for storage account"
  value       = azurerm_storage_account.main.primary_access_key
  sensitive   = true
}

output "storage_account_primary_blob_endpoint" {
  description = "Primary blob endpoint"
  value       = azurerm_storage_account.main.primary_blob_endpoint
}

output "key_vault_id" {
  description = "ID of the Key Vault"
  value       = azurerm_key_vault.main.id
}

output "key_vault_name" {
  description = "Name of the Key Vault"
  value       = azurerm_key_vault.main.name
}

output "key_vault_uri" {
  description = "URI of the Key Vault"
  value       = azurerm_key_vault.main.vault_uri
}

output "container_registry_id" {
  description = "ID of the Container Registry"
  value       = azurerm_container_registry.main.id
}

output "container_registry_name" {
  description = "Name of the Container Registry"
  value       = azurerm_container_registry.main.name
}

output "container_registry_login_server" {
  description = "Login server for Container Registry"
  value       = azurerm_container_registry.main.login_server
}
