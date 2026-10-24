resource "pinecone_index" "veritas" {
  name      = var.pinecone_index_name
  dimension = 1024
  metric    = "cosine"

  spec = {
    serverless = {
      cloud  = var.pinecone_cloud
      region = var.pinecone_region
    }
  }
}
