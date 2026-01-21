terraform {
  required_version = ">= 1.5.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.85"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
    postgresql = {
      source  = "cyrilgdn/postgresql"
      version = "~> 1.21"
    }
  }
}

provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy    = false
      recover_soft_deleted_key_vaults = true
    }
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
  }
}

# PostgreSQL provider - configured after database module creates the server
# Uses module outputs for connection details
provider "postgresql" {
  host            = module.database.server_fqdn
  port            = 5432
  database        = "postgres"
  username        = var.database_admin_username
  password        = module.database.admin_password
  sslmode         = "require"
  connect_timeout = 15
  superuser       = false
}
