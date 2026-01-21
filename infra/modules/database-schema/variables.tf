variable "server_fqdn" {
  description = "FQDN of the PostgreSQL server"
  type        = string
}

variable "database_name" {
  description = "Name of the database"
  type        = string
}

variable "admin_username" {
  description = "Administrator username"
  type        = string
}

variable "admin_password" {
  description = "Administrator password"
  type        = string
  sensitive   = true
}
