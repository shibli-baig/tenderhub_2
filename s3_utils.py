"""
S3 Utility Functions for TenderHub
==================================
Reusable functions for S3 file operations.

Features:
- Upload files to S3
- Download files from S3
- Generate presigned URLs for temporary access
- Delete files from S3
- List files in S3
- Copy files within S3

Usage:
    from s3_utils import upload_to_s3, get_presigned_url, delete_from_s3
    
    # Upload a file
    success, url = upload_to_s3(file_data, 'tenders/123/document.pdf', 'application/pdf')
    
    # Get temporary download URL (expires in 1 hour)
    download_url = get_presigned_url('tenders/123/document.pdf', expiration=3600)
    
    # Delete a file
    success = delete_from_s3('tenders/123/old_document.pdf')
"""

import os
import boto3
from botocore.exceptions import ClientError
from typing import Dict, List, Tuple, Optional, BinaryIO
from datetime import datetime, timedelta
import mimetypes
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Environment configuration
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_S3_BUCKET_NAME = os.getenv('AWS_S3_BUCKET_NAME', 'tenderhub-documents')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
USE_S3_STORAGE = os.getenv('USE_S3_STORAGE', 'true').lower() == 'true'

# Initialize S3 client
_s3_client = None


def get_s3_client():
    """Get or create S3 client singleton"""
    global _s3_client
    
    if _s3_client is None:
        if not USE_S3_STORAGE:
            raise ValueError("S3 storage is disabled. Set USE_S3_STORAGE=true in environment")
        
        if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
            raise ValueError(
                "AWS credentials not configured. "
                "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables"
            )
        
        _s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
    
    return _s3_client


def upload_to_s3(
    file_data: bytes,
    s3_key: str,
    content_type: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None,
    make_public: bool = False
) -> Tuple[bool, str]:
    """
    Upload file data to S3
    
    Args:
        file_data: Binary file data to upload
        s3_key: S3 object key (path) e.g. 'tenders/123/document.pdf'
        content_type: MIME type (auto-detected if not provided)
        metadata: Optional metadata dict (must be string key-value pairs)
        make_public: If True, file will be publicly readable
    
    Returns:
        Tuple of (success: bool, url_or_error: str)
        - On success: (True, 'https://bucket.s3.region.amazonaws.com/key')
        - On failure: (False, 'error message')
    
    Example:
        success, url = upload_to_s3(
            file_data=pdf_bytes,
            s3_key='tenders/123/document.pdf',
            content_type='application/pdf',
            metadata={'tender_id': '123', 'document_type': 'pdf'}
        )
    """
    try:
        s3_client = get_s3_client()
        
        # Auto-detect content type if not provided
        if not content_type:
            content_type, _ = mimetypes.guess_type(s3_key)
            if not content_type:
                content_type = 'application/octet-stream'
        
        # Prepare extra args
        extra_args = {'ContentType': content_type}
        
        if metadata:
            # Convert all metadata values to strings
            extra_args['Metadata'] = {k: str(v) for k, v in metadata.items()}
        
        # Note: ACLs are disabled on this bucket. Use presigned URLs for access instead.
        # if make_public:
        #     extra_args['ACL'] = 'public-read'  # Disabled - bucket doesn't allow ACLs
        
        # Upload to S3
        s3_client.put_object(
            Bucket=AWS_S3_BUCKET_NAME,
            Key=s3_key,
            Body=file_data,
            **extra_args
        )
        
        # Generate URL
        url = f"https://{AWS_S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        return True, url
    
    except ClientError as e:
        logger.error(f"S3 upload error: {e}")
        return False, f"S3 upload error: {e}"
    except Exception as e:
        logger.error(f"Unexpected error during S3 upload: {e}")
        return False, f"Unexpected error: {e}"


def upload_fileobj_to_s3(
    file_obj: BinaryIO,
    s3_key: str,
    content_type: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None
) -> Tuple[bool, str]:
    """
    Upload file object (stream) to S3
    
    This is more memory-efficient for large files as it streams the data.
    
    Args:
        file_obj: File-like object (e.g., from UploadFile.file in FastAPI)
        s3_key: S3 object key
        content_type: MIME type
        metadata: Optional metadata
    
    Returns:
        Tuple of (success: bool, url_or_error: str)
    
    Example:
        @app.post("/upload")
        async def upload_file(file: UploadFile):
            success, url = upload_fileobj_to_s3(
                file.file,
                f"uploads/{file.filename}",
                file.content_type
            )
    """
    try:
        s3_client = get_s3_client()
        
        if not content_type:
            content_type, _ = mimetypes.guess_type(s3_key)
            if not content_type:
                content_type = 'application/octet-stream'
        
        extra_args = {'ContentType': content_type}
        if metadata:
            extra_args['Metadata'] = {k: str(v) for k, v in metadata.items()}
        
        s3_client.upload_fileobj(
            file_obj,
            AWS_S3_BUCKET_NAME,
            s3_key,
            ExtraArgs=extra_args
        )
        
        url = f"https://{AWS_S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        return True, url
    
    except ClientError as e:
        logger.error(f"S3 upload error: {e}")
        return False, f"S3 upload error: {e}"
    except Exception as e:
        logger.error(f"Unexpected error during S3 upload: {e}")
        return False, f"Unexpected error: {e}"


def get_s3_stream(s3_key: str):
    """
    Get streaming response from S3 for large files
    
    Args:
        s3_key: S3 object key
    
    Returns:
        S3 object Body stream or None if error
    
    Example:
        from fastapi.responses import StreamingResponse
        
        @app.get("/download/{s3_key:path}")
        def download_file(s3_key: str):
            stream = get_s3_stream(s3_key)
            if stream:
                return StreamingResponse(
                    stream,
                    media_type='application/octet-stream'
                )
            else:
                raise HTTPException(404, "File not found")
    """
    try:
        s3_client = get_s3_client()
        response = s3_client.get_object(
            Bucket=AWS_S3_BUCKET_NAME,
            Key=s3_key
        )
        return response['Body']
    except ClientError as e:
        logger.error(f"Error getting S3 stream: {e}")
        return None


def download_from_s3(s3_key: str) -> Tuple[bool, bytes, str]:
    """
    Download file from S3
    
    Args:
        s3_key: S3 object key
    
    Returns:
        Tuple of (success: bool, file_data: bytes, error_or_content_type: str)
    
    Example:
        success, data, content_type = download_from_s3('tenders/123/document.pdf')
        if success:
            return Response(data, media_type=content_type)
    """
    try:
        s3_client = get_s3_client()
        
        response = s3_client.get_object(
            Bucket=AWS_S3_BUCKET_NAME,
            Key=s3_key
        )
        
        file_data = response['Body'].read()
        content_type = response.get('ContentType', 'application/octet-stream')
        
        return True, file_data, content_type
    
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.error(f"File not found in S3: {s3_key}")
            return False, b'', f"File not found: {s3_key}"
        logger.error(f"S3 download error: {e}")
        return False, b'', f"S3 download error: {e}"
    except Exception as e:
        logger.error(f"Unexpected error during S3 download: {e}")
        return False, b'', f"Unexpected error: {e}"


def get_presigned_url(
    s3_key: str,
    expiration: int = 3600,
    filename: Optional[str] = None
) -> Optional[str]:
    """
    Generate presigned URL for temporary file access
    
    Presigned URLs allow temporary access to private S3 objects without
    making them public. Useful for secure file downloads.
    
    Args:
        s3_key: S3 object key
        expiration: URL expiration time in seconds (default: 1 hour)
        filename: Optional filename for Content-Disposition header
    
    Returns:
        Presigned URL string or None if error
    
    Example:
        # Generate 1-hour download link
        download_url = get_presigned_url('tenders/123/document.pdf', expiration=3600)
        
        # Force download with specific filename
        download_url = get_presigned_url(
            'tenders/123/abc.pdf',
            filename='Tender_Document.pdf'
        )
    """
    try:
        s3_client = get_s3_client()
        
        params = {
            'Bucket': AWS_S3_BUCKET_NAME,
            'Key': s3_key
        }
        
        # Add filename for download
        if filename:
            params['ResponseContentDisposition'] = f'attachment; filename="{filename}"'
        
        url = s3_client.generate_presigned_url(
            'get_object',
            Params=params,
            ExpiresIn=expiration
        )
        
        logger.debug(f"Generated presigned URL for {s3_key} (expires in {expiration}s)")
        return url
    
    except ClientError as e:
        logger.error(f"Error generating presigned URL: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error generating presigned URL: {e}")
        return None


def delete_from_s3(s3_key: str) -> bool:
    """
    Delete file from S3
    
    Args:
        s3_key: S3 object key
    
    Returns:
        True if successful, False otherwise
    
    Example:
        if delete_from_s3('tenders/123/old_document.pdf'):
            print("File deleted successfully")
    """
    try:
        s3_client = get_s3_client()
        
        s3_client.delete_object(
            Bucket=AWS_S3_BUCKET_NAME,
            Key=s3_key
        )
        
        logger.info(f"Deleted file from S3: {s3_key}")
        return True
    
    except ClientError as e:
        logger.error(f"Error deleting from S3: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error deleting from S3: {e}")
        return False


def file_exists_in_s3(s3_key: str) -> bool:
    """
    Check if file exists in S3
    
    Args:
        s3_key: S3 object key
    
    Returns:
        True if file exists, False otherwise
    """
    try:
        s3_client = get_s3_client()
        s3_client.head_object(Bucket=AWS_S3_BUCKET_NAME, Key=s3_key)
        return True
    except ClientError:
        return False
    except Exception as e:
        logger.error(f"Error checking file existence in S3: {e}")
        return False


def list_s3_files(prefix: str, max_results: int = 1000) -> List[Dict]:
    """
    List files in S3 with given prefix
    
    Args:
        prefix: S3 key prefix (folder path) e.g. 'tenders/123/'
        max_results: Maximum number of results to return
    
    Returns:
        List of dicts with file information:
        [
            {
                'key': 'tenders/123/document.pdf',
                'size': 1024000,
                'last_modified': datetime(...),
                'etag': '...'
            },
            ...
        ]
    
    Example:
        # List all files for a tender
        files = list_s3_files('tenders/123/')
        for file in files:
            print(f"{file['key']} - {file['size']} bytes")
    """
    try:
        s3_client = get_s3_client()
        
        response = s3_client.list_objects_v2(
            Bucket=AWS_S3_BUCKET_NAME,
            Prefix=prefix,
            MaxKeys=max_results
        )
        
        files = []
        for obj in response.get('Contents', []):
            files.append({
                'key': obj['Key'],
                'size': obj['Size'],
                'last_modified': obj['LastModified'],
                'etag': obj['ETag']
            })
        
        return files
    
    except ClientError as e:
        logger.error(f"Error listing S3 files: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error listing S3 files: {e}")
        return []


def copy_s3_file(source_key: str, destination_key: str) -> bool:
    """
    Copy file within S3
    
    Args:
        source_key: Source S3 object key
        destination_key: Destination S3 object key
    
    Returns:
        True if successful, False otherwise
    
    Example:
        # Copy file to backup
        copy_s3_file(
            'tenders/123/document.pdf',
            'tenders/123/backup/document.pdf'
        )
    """
    try:
        s3_client = get_s3_client()
        
        copy_source = {
            'Bucket': AWS_S3_BUCKET_NAME,
            'Key': source_key
        }
        
        s3_client.copy_object(
            CopySource=copy_source,
            Bucket=AWS_S3_BUCKET_NAME,
            Key=destination_key
        )
        
        return True
    
    except ClientError as e:
        logger.error(f"Error copying S3 file: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error copying S3 file: {e}")
        return False


def get_file_info(s3_key: str) -> Optional[Dict]:
    """
    Get file metadata from S3
    
    Args:
        s3_key: S3 object key
    
    Returns:
        Dict with file information or None if error:
        {
            'size': 1024000,
            'content_type': 'application/pdf',
            'last_modified': datetime(...),
            'metadata': {...}
        }
    """
    try:
        s3_client = get_s3_client()
        
        response = s3_client.head_object(
            Bucket=AWS_S3_BUCKET_NAME,
            Key=s3_key
        )
        
        return {
            'size': response['ContentLength'],
            'content_type': response['ContentType'],
            'last_modified': response['LastModified'],
            'metadata': response.get('Metadata', {})
        }
    
    except ClientError as e:
        logger.error(f"Error getting file info: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting file info: {e}")
        return None


def generate_s3_key(category: str, identifier: str, filename: str) -> str:
    """
    Generate standardized S3 key with folder structure
    
    Args:
        category: Category/type (e.g., 'tenders', 'tasks', 'reports')
        identifier: Unique identifier (e.g., tender_id, task_id)
        filename: Original filename
    
    Returns:
        S3 key string
    
    Example:
        key = generate_s3_key('tenders', '12345', 'Tender Document.pdf')
        # Returns: 'tenders/12345/Tender_Document.pdf'
    """
    # Sanitize filename
    safe_filename = "".join(c for c in filename if c.isalnum() or c in '._- ')
    safe_filename = safe_filename.replace(' ', '_')
    
    return f"{category}/{identifier}/{safe_filename}"


def get_public_url(s3_key: str) -> str:
    """
    Get public URL for S3 object
    
    Note: This returns the URL but the file must be publicly accessible
    for this URL to work. Use get_presigned_url() for private files.
    
    Args:
        s3_key: S3 object key
    
    Returns:
        Public URL string
    """
    return f"https://{AWS_S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"


# Convenience function for FastAPI route handlers
async def handle_file_upload(
    file_obj: BinaryIO,
    category: str,
    identifier: str,
    filename: str,
    content_type: Optional[str] = None,
    metadata: Optional[Dict] = None
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Handle file upload with standardized error handling
    
    Args:
        file_obj: File object from request
        category: File category (e.g., 'tenders', 'tasks')
        identifier: Related entity ID
        filename: Original filename
        content_type: MIME type
        metadata: Optional metadata
    
    Returns:
        Tuple of (success: bool, s3_key: Optional[str], url_or_error: Optional[str])
    
    Example:
        @app.post("/upload/tender/{tender_id}")
        async def upload_tender_doc(tender_id: str, file: UploadFile):
            success, s3_key, url = await handle_file_upload(
                file.file,
                'tenders',
                tender_id,
                file.filename,
                file.content_type
            )
            if success:
                return {"url": url, "s3_key": s3_key}
            else:
                raise HTTPException(500, url)  # url contains error message
    """
    try:
        # Generate S3 key
        s3_key = generate_s3_key(category, identifier, filename)
        
        # Upload to S3
        success, result = upload_fileobj_to_s3(
            file_obj,
            s3_key,
            content_type,
            metadata
        )
        
        if success:
            return True, s3_key, result  # result is URL
        else:
            return False, None, result  # result is error message
    
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        return False, None, f"File upload failed: {e}"

