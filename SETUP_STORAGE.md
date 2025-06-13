# Quick Setup Guide for DigitalOcean Spaces Storage

This guide will help you quickly set up DigitalOcean Spaces storage for your football analysis project.

## 🚀 Quick Start (5 minutes)

### Step 1: Install Dependencies
```bash
pip install boto3 python-dotenv
```

### Step 2: Set Up Your Credentials

**Option A: Using .env file (Recommended)**
```bash
# Copy the example file
cp .env.example .env

# Edit .env with your actual credentials
nano .env
```

Edit the `.env` file:
```bash
DO_SPACES_ACCESS_KEY_ID=your_actual_access_key_id
DO_SPACES_SECRET_ACCESS_KEY=your_actual_secret_access_key
DO_SPACES_BUCKET=your_actual_bucket_name
DO_SPACES_REGION=nyc3
```

**Option B: Using environment variables**
```bash
export DO_SPACES_ACCESS_KEY_ID="your_access_key_id"
export DO_SPACES_SECRET_ACCESS_KEY="your_secret_access_key"
export DO_SPACES_BUCKET="your_bucket_name"
```

### Step 3: Run Analysis with Upload
```bash
# Basic usage - uploads CSV files and video
python main.py --input your_video.mp4 --upload-to-spaces

# CSV only (faster) - uploads only statistics files
python main.py --input your_video.mp4 --upload-to-spaces --upload-csv-only

# Memory-efficient for large videos
python main.py --input large_video.mp4 --memory-efficient --upload-to-spaces
```

## 📁 What Gets Uploaded

Your files will be organized in your DigitalOcean Spaces bucket like this:

```
your_bucket/
├── football_analysis/
│   ├── csv/
│   │   ├── video_name_team_stats.csv
│   │   └── video_name_player_stats.csv
│   └── videos/
│       └── video_name_output.avi
```

## 🔧 Getting DigitalOcean Spaces Credentials

1. **Log in to DigitalOcean**: Go to [DigitalOcean](https://cloud.digitalocean.com/)

2. **Create a Space**: 
   - Navigate to "Spaces" in the sidebar
   - Click "Create a Space"
   - Choose a region (e.g., NYC3)
   - Give it a name (this is your bucket name)

3. **Generate API Keys**:
   - Go to "API" in the sidebar
   - Click "Generate New Key" under "Spaces access keys"
   - Copy the Access Key ID and Secret Access Key

4. **Update your .env file** with the actual values

## ✅ Test Your Setup

Run the example script to test your configuration:
```bash
python example_storage_usage.py
```

You should see:
- ✅ Connected to DigitalOcean Spaces
- ✅ Uploader created successfully
- ✅ Connection test successful

## 🛠️ Troubleshooting

**"Failed to connect to DigitalOcean Spaces"**
- Check your access key ID and secret access key
- Verify the bucket name exists
- Ensure the region is correct

**"Access denied to bucket"**
- Check bucket permissions
- Verify API key has read/write access

**"Credentials not found"**
- Make sure .env file exists and has correct format
- Check environment variables are set
- Verify no typos in variable names

## 🔒 Security Notes

- ✅ .env files are already in .gitignore (won't be committed)
- ✅ Use .env files for local development
- ✅ Use environment variables for production/CI
- ✅ Never commit credentials to version control

## 📊 Usage Examples

**Basic analysis with upload:**
```bash
python main.py --input match.mp4 --upload-to-spaces
```

**Fast CSV-only upload:**
```bash
python main.py --input match.mp4 --upload-to-spaces --upload-csv-only
```

**Custom folder organization:**
```bash
python main.py --input match.mp4 --upload-to-spaces --spaces-folder-prefix "season_2024/matches"
```

**Memory-efficient for large files:**
```bash
python main.py --input large_match.mp4 --memory-efficient --batch-size 30 --upload-to-spaces
```

## 💰 Cost Estimation

DigitalOcean Spaces pricing (as of 2024):
- **Storage**: $5/month for 250 GB
- **Bandwidth**: $10/TB for outbound transfer

Typical costs for football analysis:
- **CSV files**: ~1-10 KB each (negligible cost)
- **Video files**: 100 MB - 2 GB each
- **Monthly cost**: Usually under $10 for regular use

## 🎯 Next Steps

1. Set up your credentials using the steps above
2. Test with a small video file first
3. Run your analysis with `--upload-to-spaces`
4. Check your DigitalOcean Spaces bucket for uploaded files
5. Share the bucket URLs with your team if needed

For detailed documentation, see `STORAGE_README.md`.
