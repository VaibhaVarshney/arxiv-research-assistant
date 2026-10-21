"""
Step 2a: Create the S3 vector bucket and index. Run once.

Usage:
    python src/indexing/setup_index.py

Safe to re-run: if they already exist, it just says so.
"""
from botocore.exceptions import ClientError

from aws_config import (EMBED_DIM, INDEX_NAME, NON_FILTERABLE_KEYS, REGION,
                        VECTOR_BUCKET, s3vectors)


def create_if_missing(label: str, fn, **kwargs):
    try:
        fn(**kwargs)
        print(f"Created {label}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConflictException":
            print(f"{label} already exists, skipping")
        else:
            raise


def main():
    print(f"Region: {REGION}")
    create_if_missing(
        f"vector bucket '{VECTOR_BUCKET}'",
        s3vectors.create_vector_bucket,
        vectorBucketName=VECTOR_BUCKET,
    )
    # Dimension, distance metric and non-filterable keys CANNOT be changed later.
    create_if_missing(
        f"index '{INDEX_NAME}'",
        s3vectors.create_index,
        vectorBucketName=VECTOR_BUCKET,
        indexName=INDEX_NAME,
        dataType="float32",
        dimension=EMBED_DIM,
        distanceMetric="cosine",
        metadataConfiguration={"nonFilterableMetadataKeys": NON_FILTERABLE_KEYS},
    )


if __name__ == "__main__":
    main()
