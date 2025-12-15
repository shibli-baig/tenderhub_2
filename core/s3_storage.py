"""
AWS S3 Storage Module for TenderHub
====================================

Handles file uploads, downloads, and deletion from AWS S3.
Provides presigned URLs for secure file access.

Author: TenderHub Team
"""

import os
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
from typing import Optional, BinaryIO
import logging
from pathlib import Path
import uuid
from datetime import timedelta

logger = logging.getLogger(__name__)

# S3 Configuration from environment variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
USE_S3_STORAGE = os.getenv("USE_S3_STORAGE", "True").lower() in ("true", "1", "yes")


class S3StorageManager:
    """Manages file storage operations with AWS S3."""
    
    def __init__(self):
        """Initialize S3 client if credentials are available."""
        self.enabled = False
        self.s3_client = None
        self.bucket_name = AWS_S3_BUCKET_NAME
        
        if not USE_S3_STORAGE:
            logger.warning("S3 storage is disabled. Files will be stored locally.")
            return
        
        if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET_NAME]):
            logger.warning(
                "S3 credentials not fully configured. Files will be stored locally. "
                "Required: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET_NAME"
            )
            return
        
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=AWS_REGION,
                config=Config(
                    signature_version='s3v4',
                    retries={'max_attempts': 3, 'mode': 'standard'}
                )
            )
            self.enabled = True
            logger.info(f"S3 storage initialized successfully. Bucket: {self.bucket_name}, Region: {AWS_REGION}")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            self.enabled = False
    
    def upload_file(
        self,
        file_obj: BinaryIO,
        file_key: str,
        content_type: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> Optional[str]:
        """
        Upload a file to S3.
        
        Args:
            file_obj: File-like object to upload
            file_key: S3 object key (path within bucket)
            content_type: MIME type of the file
            metadata: Optional metadata dictionary
        
        Returns:
            S3 URL of uploaded file, or None if upload failed
        """
        if not self.enabled:
            logger.warning("S3 storage not enabled. Cannot upload file.")
            return None
        
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
            if metadata:
                extra_args['Metadata'] = metadata
            
            # Add server-side encryption
            extra_args['ServerSideEncryption'] = 'AES256'
            
            self.s3_client.upload_fileobj(
                file_obj,
                self.bucket_name,
                file_key,
                ExtraArgs=extra_args
            )
            
            # Return the S3 URL
            s3_url = f"s3://{self.bucket_name}/{file_key}"
            logger.info(f"File uploaded successfully: {s3_url}")
            return s3_url
        
        except ClientError as e:
            logger.error(f"Failed to upload file to S3: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during S3 upload: {e}")
            return None
    
    def download_file(self, file_key: str, destination_path: str) -> bool:
        """
        Download a file from S3 to local filesystem.
        
        Args:
            file_key: S3 object key
            destination_path: Local path to save the file
        
        Returns:
            True if download successful, False otherwise
        """
        if not self.enabled:
            logger.warning("S3 storage not enabled. Cannot download file.")
            return False
        
        try:
            self.s3_client.download_file(self.bucket_name, file_key, destination_path)
            logger.info(f"File downloaded successfully: {file_key} -> {destination_path}")
            return True
        except ClientError as e:
            logger.error(f"Failed to download file from S3: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during S3 download: {e}")
            return False
    
    def delete_file(self, file_key: str) -> bool:
        """
        Delete a file from S3.
        
        Args:
            file_key: S3 object key to delete
        
        Returns:
            True if deletion successful, False otherwise
        """
        if not self.enabled:
            logger.warning("S3 storage not enabled. Cannot delete file.")
            return False
        
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=file_key)
            logger.info(f"File deleted successfully: {file_key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete file from S3: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during S3 deletion: {e}")
            return False
    
    def generate_presigned_url(
        self,
        file_key: str,
        expiration: int = 3600,
        operation: str = 'get_object'
    ) -> Optional[str]:
        """
        Generate a presigned URL for temporary access to a file.
        
        Args:
            file_key: S3 object key
            expiration: URL expiration time in seconds (default: 1 hour)
            operation: S3 operation (default: 'get_object')
        
        Returns:
            Presigned URL, or None if generation failed
        """
        if not self.enabled:
            logger.warning("S3 storage not enabled. Cannot generate presigned URL.")
            return None
        
        try:
            url = self.s3_client.generate_presigned_url(
                operation,
                Params={'Bucket': self.bucket_name, 'Key': file_key},
                ExpiresIn=expiration
            )
            logger.debug(f"Generated presigned URL for {file_key} (expires in {expiration}s)")
            return url
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error generating presigned URL: {e}")
            return None
    
    def file_exists(self, file_key: str) -> bool:
        """
        Check if a file exists in S3.
        
        Args:
            file_key: S3 object key
        
        Returns:
            True if file exists, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=file_key)
            return True
        except ClientError:
            return False
        except Exception as e:
            logger.error(f"Error checking file existence: {e}")
            return False
    
    def get_file_url(self, file_key: str, expiration: int = 3600) -> Optional[str]:
        """
        Get a URL for accessing a file (presigned URL for private files).
        
        Args:
            file_key: S3 object key or full S3 URL
            expiration: URL expiration time in seconds
        
        Returns:
            Presigned URL for file access
        """
        # Extract key from s3:// URL if needed
        if file_key.startswith("s3://"):
            # Format: s3://bucket-name/path/to/file
            parts = file_key.replace("s3://", "").split("/", 1)
            if len(parts) == 2:
                file_key = parts[1]
        
        return self.generate_presigned_url(file_key, expiration)
    
    @staticmethod
    def generate_unique_key(user_id: str, filename: str, folder: str = "uploads") -> str:
        """
        Generate a unique S3 key for a file.
        
        Args:
            user_id: User ID for organization
            filename: Original filename
            folder: Top-level folder (e.g., 'certificates', 'projects')
        
        Returns:
            Unique S3 key
        """
        file_extension = Path(filename).suffix
        unique_id = uuid.uuid4().hex
        return f"{folder}/{user_id}/{unique_id}{file_extension}"


# Global S3 manager instance
s3_manager = S3StorageManager()


# Convenience functions for easier usage
def upload_to_s3(file_obj: BinaryIO, user_id: str, filename: str, 
                 folder: str = "uploads", content_type: Optional[str] = None) -> Optional[str]:
    """
    Convenience function to upload a file to S3.
    
    Returns:
        S3 URL of uploaded file
    """
    file_key = S3StorageManager.generate_unique_key(user_id, filename, folder)
    return s3_manager.upload_file(file_obj, file_key, content_type)


def get_s3_file_url(s3_url: str, expiration: int = 3600) -> Optional[str]:
    """
    Convenience function to get a presigned URL for an S3 file.
    
    Returns:
        Presigned URL for file access
    """
    return s3_manager.get_file_url(s3_url, expiration)


def delete_from_s3(s3_url: str) -> bool:
    """
    Convenience function to delete a file from S3.
    
    Returns:
        True if deletion successful
    """
    # Extract key from s3:// URL
    if s3_url.startswith("s3://"):
        parts = s3_url.replace("s3://", "").split("/", 1)
        if len(parts) == 2:
            file_key = parts[1]
            return s3_manager.delete_file(file_key)
    return False
