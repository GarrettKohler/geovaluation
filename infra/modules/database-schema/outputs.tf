output "schema_name" {
  description = "Name of the created schema"
  value       = postgresql_schema.dooh.name
}
