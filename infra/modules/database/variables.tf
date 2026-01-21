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

variable "sku_name" {
  description = "SKU for PostgreSQL Flexible Server"
  type        = string
  default     = "B_Standard_B1ms"
}

variable "storage_mb" {
  description = "Storage size in MB"
  type        = number
  default     = 32768
}

variable "admin_username" {
  description = "Administrator username"
  type        = string
  default     = "pgadmin"
}

variable "key_vault_id" {
  description = "ID of Key Vault for storing secrets"
  type        = string
}
