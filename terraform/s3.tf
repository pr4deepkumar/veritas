resource "aws_s3_bucket" "documents" {
  bucket        = "${var.app_name}-${var.s3_bucket_name}-${var.environment}"
  force_destroy = var.s3_force_destroy
}

resource "aws_s3_bucket_server_side_encryption_configuration" "documents_crypto" {
  bucket = aws_s3_bucket.documents.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "documents_privacy" {
  bucket = aws_s3_bucket.documents.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Create folder placeholders inside the S3 bucket for input PDFs and parsed JSON outputs
resource "aws_s3_object" "input_pdfs_folder" {
  bucket = aws_s3_bucket.documents.id
  key    = "input_pdfs/"
}

resource "aws_s3_object" "parsed_json_folder" {
  bucket = aws_s3_bucket.documents.id
  key    = "parsed_json/"
}
