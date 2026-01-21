output "server_id" {
  description = "ID of the PostgreSQL server"
  value       = azurerm_postgresql_flexible_server.main.id
}

output "server_name" {
  description = "Name of the PostgreSQL server"
  value       = azurerm_postgresql_flexible_server.main.name
}

output "server_fqdn" {
  description = "FQDN of the PostgreSQL server"
  value       = azurerm_postgresql_flexible_server.main.fqdn
}

output "database_name" {
  description = "Name of the main database"
  value       = azurerm_postgresql_flexible_server_database.dooh.name
}

output "admin_username" {
  description = "Administrator username"
  value       = var.admin_username
}

output "admin_password" {
  description = "Administrator password"
  value       = random_password.postgres.result
  sensitive   = true
}
