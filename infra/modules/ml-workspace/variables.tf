variable "resource_group_name" {
  description = "Name of the resource group"
  type        = string
}

variable "location" {
  description = "Azure region"
  type        = string
}

variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "random_suffix" {
  description = "Random suffix for globally unique names"
  type        = string
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

variable "storage_account_id" {
  description = "ID of the storage account for ML workspace"
  type        = string
}

variable "key_vault_id" {
  description = "ID of the Key Vault for ML workspace"
  type        = string
}

variable "container_registry_id" {
  description = "ID of the Container Registry for ML workspace"
  type        = string
}

variable "sku_name" {
  description = "SKU for ML Workspace (Basic or Enterprise)"
  type        = string
  default     = "Basic"
}
