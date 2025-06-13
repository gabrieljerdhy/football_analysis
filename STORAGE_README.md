# DigitalOcean Spaces Storage Integration

This document explains how to use the DigitalOcean Spaces storage integration with the football analysis project. This feature allows you to automatically upload CSV statistics files and output videos to DigitalOcean Spaces (S3-compatible storage).

## Features

- ✅ Automatic upload of CSV statistics files (team and player stats)
- ✅ Automatic upload of output videos
- ✅ Support for both regular and memory-efficient processing modes
- ✅ Configurable folder organization in the bucket
- ✅ Option to upload only CSV files (skip video for faster processing)
- ✅ Environment variable and command-line configuration
- ✅ Connection testing and error handling
- ✅ Upload progress feedback

## Prerequisites

1. **Install required dependencies** (if not already installed):
   ```bash
   pip install boto3 python-dotenv
   ```

2. **DigitalOcean Spaces Account**:
   - Create a DigitalOcean Spaces bucket
   - Generate API keys (Access Key ID and Secret Access Key)

## Configuration Methods

### Method 1: .env File (Recommended)

1. **Copy the example file**:
   ```bash
   cp .env.example .env
   ```

2. **Edit the .env file** with your actual credentials:
   ```bash
   # .env file
   DO_SPACES_ACCESS_KEY_ID=your_actual_access_key_id
   DO_SPACES_SECRET_ACCESS_KEY=your_actual_secret_access_key
   DO_SPACES_BUCKET=your_actual_bucket_name
   DO_SPACES_REGION=nyc3
   DO_SPACES_FOLDER_PREFIX=football_analysis
   ```

3. **Run with the upload flag**:
   ```bash
   python main.py --input video.mp4 --upload-to-spaces
   ```

### Method 2: Environment Variables

Set the following environment variables:

```bash
export DO_SPACES_ACCESS_KEY_ID="your_access_key_id"
export DO_SPACES_SECRET_ACCESS_KEY="your_secret_access_key"
export DO_SPACES_BUCKET="your_bucket_name"
export DO_SPACES_REGION="nyc3"  # Optional, defaults to nyc3
```

Then run with the `--upload-to-spaces` flag:

```bash
python main.py --input video.mp4 --upload-to-spaces
```

### Method 3: Command Line Arguments

```bash
python main.py --input video.mp4 \
    --upload-to-spaces \
    --spaces-access-key-id "your_access_key_id" \
    --spaces-secret-access-key "your_secret_access_key" \
    --spaces-bucket "your_bucket_name" \
    --spaces-region "nyc3"
```

## Usage Examples

### Basic Upload (CSV + Video)

```bash
python main.py --input football_match.mp4 --upload-to-spaces
```

### Upload Only CSV Files (Faster)

```bash
python main.py --input football_match.mp4 --upload-to-spaces --upload-csv-only
```

### Memory-Efficient Processing with Upload

```bash
python main.py --input large_video.mp4 \
    --memory-efficient \
    --batch-size 30 \
    --upload-to-spaces
```

### Custom Folder Organization

```bash
python main.py --input video.mp4 \
    --upload-to-spaces \
    --spaces-folder-prefix "season_2024/match_analysis"
```

## File Organization

Files are organized in your DigitalOcean Spaces bucket as follows:

```
your_bucket/
├── football_analysis/           # Default folder prefix
│   ├── csv/
│   │   ├── video_name_team_stats.csv
│   │   └── video_name_player_stats.csv
│   └── videos/
│       └── video_name_output.avi
```

With custom folder prefix:

```
your_bucket/
├── your_custom_prefix/
│   ├── csv/
│   │   ├── video_name_team_stats.csv
│   │   └── video_name_player_stats.csv
│   └── videos/
│       └── video_name_output.avi
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--upload-to-spaces` | Enable upload to DigitalOcean Spaces | False |
| `--spaces-access-key-id` | DigitalOcean Spaces access key ID | From env var |
| `--spaces-secret-access-key` | DigitalOcean Spaces secret access key | From env var |
| `--spaces-bucket` | Bucket name | From env var |
| `--spaces-region` | DigitalOcean region | nyc3 |
| `--spaces-folder-prefix` | Folder prefix in bucket | football_analysis |
| `--upload-csv-only` | Upload only CSV files, skip video | False |

## Error Handling

The system includes comprehensive error handling:

- **Connection Testing**: Verifies bucket access before processing
- **Upload Validation**: Checks file existence before upload
- **Graceful Degradation**: Continues processing even if upload fails
- **Progress Feedback**: Shows upload status and file sizes

## Security Best Practices

1. **Use Environment Variables**: Store credentials in environment variables, not in code
2. **Limit Permissions**: Create API keys with minimal required permissions
3. **Bucket Access**: Ensure your bucket has appropriate access controls
4. **Key Rotation**: Regularly rotate your API keys

## Troubleshooting

### Common Issues

1. **"boto3 not available"**
   ```bash
   pip install boto3
   ```

2. **"Failed to connect to DigitalOcean Spaces"**
   - Check your access key ID and secret access key
   - Verify the bucket name exists
   - Ensure the region is correct

3. **"Access denied to bucket"**
   - Check bucket permissions
   - Verify API key has read/write access to the bucket

4. **"No bucket name provided"**
   - Set the `DO_SPACES_BUCKET` environment variable, or
   - Use the `--spaces-bucket` command line argument

### Debug Mode

For detailed debugging, you can check the connection manually:

```python
from utils.storage_utils import create_uploader_from_env

uploader = create_uploader_from_env()
if uploader:
    success = uploader.test_connection('your_bucket_name')
    print(f"Connection test: {'✅ Success' if success else '❌ Failed'}")
```

## Integration with Existing Workflows

The storage functionality integrates seamlessly with existing features:

- **Memory-Efficient Processing**: Works with `--memory-efficient` flag
- **Goal Detection**: Uploads enhanced statistics with goal detection data
- **Jersey Number Detection**: Includes jersey number data in uploaded CSVs
- **Custom Goals**: Works with manual goal configuration files

## Performance Considerations

- **CSV Upload**: Very fast (typically < 1 second per file)
- **Video Upload**: Depends on video size and internet connection
- **Memory Usage**: Upload process uses minimal additional memory
- **Parallel Processing**: Uploads happen after video processing is complete

## Cost Considerations

DigitalOcean Spaces pricing (as of 2024):
- Storage: $5/month for 250 GB
- Bandwidth: $10/TB for outbound transfer
- API Requests: Included in base price

For typical football analysis:
- CSV files: ~1-10 KB each (negligible cost)
- Video files: 100 MB - 2 GB (main cost factor)

## Example Integration Script

See `example_storage_usage.py` for detailed examples of:
- Environment variable configuration
- Direct API usage
- Custom folder organization
- Error handling patterns

## Support

For issues related to:
- **DigitalOcean Spaces**: Check DigitalOcean documentation
- **boto3**: Check AWS boto3 documentation (S3-compatible)
- **This Integration**: Check the project's main README or create an issue
