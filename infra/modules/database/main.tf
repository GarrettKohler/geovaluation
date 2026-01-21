# Database Module
# Creates: PostgreSQL Flexible Server with PostGIS extension

# Generate secure password
resource "random_password" "postgres" {
  length           = 24
  special          = true
  override_special = "!#$%&*()-_=+[]{}<>:?"
}

# Store password in Key Vault
resource "azurerm_key_vault_secret" "postgres_password" {
  name         = "postgres-admin-password"
  value        = random_password.postgres.result
  key_vault_id = var.key_vault_id
}

# PostgreSQL Flexible Server
resource "azurerm_postgresql_flexible_server" "main" {
  name                   = "psql-${var.name_prefix}-${var.random_suffix}"
  resource_group_name    = var.resource_group_name
  location               = var.location
  version                = "16"
  administrator_login    = var.admin_username
  administrator_password = random_password.postgres.result

  storage_mb = var.storage_mb
  sku_name   = var.sku_name

  backup_retention_days        = 7
  geo_redundant_backup_enabled = false # Enable for prod

  tags = var.tags

  lifecycle {
    ignore_changes = [
      zone, # Availability zone is auto-assigned
    ]
  }
}

# Firewall rule - Allow Azure services (for dev)
resource "azurerm_postgresql_flexible_server_firewall_rule" "azure_services" {
  name             = "AllowAzureServices"
  server_id        = azurerm_postgresql_flexible_server.main.id
  start_ip_address = "0.0.0.0"
  end_ip_address   = "0.0.0.0"
}

# Firewall rule - Allow all (for dev only, remove for prod)
resource "azurerm_postgresql_flexible_server_firewall_rule" "allow_all" {
  name             = "AllowAll-Dev"
  server_id        = azurerm_postgresql_flexible_server.main.id
  start_ip_address = "0.0.0.0"
  end_ip_address   = "255.255.255.255"
}

# Main application database
resource "azurerm_postgresql_flexible_server_database" "dooh" {
  name      = "dooh_sites"
  server_id = azurerm_postgresql_flexible_server.main.id
  charset   = "UTF8"
  collation = "en_US.utf8"
}

# Enable PostGIS extension
resource "azurerm_postgresql_flexible_server_configuration" "extensions" {
  name      = "azure.extensions"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "POSTGIS,POSTGIS_TOPOLOGY,UUID-OSSP"
}

# Connection string secret
resource "azurerm_key_vault_secret" "postgres_connection_string" {
  name         = "postgres-connection-string"
  key_vault_id = var.key_vault_id
  value        = "postgresql://${var.admin_username}:${random_password.postgres.result}@${azurerm_postgresql_flexible_server.main.fqdn}:5432/${azurerm_postgresql_flexible_server_database.dooh.name}?sslmode=require"

  depends_on = [azurerm_postgresql_flexible_server_database.dooh]
}
