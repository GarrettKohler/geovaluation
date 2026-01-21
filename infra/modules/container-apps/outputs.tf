output "environment_id" {
  description = "ID of the Container Apps Environment"
  value       = azurerm_container_app_environment.main.id
}

output "environment_name" {
  description = "Name of the Container Apps Environment"
  value       = azurerm_container_app_environment.main.name
}

output "default_domain" {
  description = "Default domain of the Container Apps Environment"
  value       = azurerm_container_app_environment.main.default_domain
}
