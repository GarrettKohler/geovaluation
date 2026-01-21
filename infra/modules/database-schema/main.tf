# Database Schema Module
# Creates: PostGIS extension, tables for DOOH site optimization

# Enable PostGIS extension in the application database
resource "postgresql_extension" "postgis" {
  name     = "postgis"
  database = var.database_name
}

resource "postgresql_extension" "uuid_ossp" {
  name     = "uuid-ossp"
  database = var.database_name
}

# Sites table - Core site data with geospatial location
resource "postgresql_schema" "dooh" {
  name     = "dooh"
  database = var.database_name
}

# Sites table - The 60k gas station advertising locations
resource "postgresql_grant" "dooh_schema_usage" {
  database    = var.database_name
  role        = var.admin_username
  schema      = postgresql_schema.dooh.name
  object_type = "schema"
  privileges  = ["CREATE", "USAGE"]
}

# Create tables using a single SQL script for proper dependency ordering
resource "null_resource" "create_tables" {
  depends_on = [
    postgresql_extension.postgis,
    postgresql_extension.uuid_ossp,
    postgresql_schema.dooh
  ]

  provisioner "local-exec" {
    command = <<-EOT
      PGPASSWORD='${var.admin_password}' psql \
        -h ${var.server_fqdn} \
        -U ${var.admin_username} \
        -d ${var.database_name} \
        -f ${path.module}/schema.sql
    EOT
  }

  triggers = {
    schema_hash = filemd5("${path.module}/schema.sql")
  }
}
