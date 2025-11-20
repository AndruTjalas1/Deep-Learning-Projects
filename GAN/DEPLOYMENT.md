# Deployment Guide

Complete instructions for deploying the DCGAN system to production using Vercel (frontend) and Railway (backend).

## Table of Contents

1. [Frontend Deployment (Vercel)](#frontend-deployment-vercel)
2. [Backend Deployment (Railway)](#backend-deployment-railway)
3. [Connecting Frontend to Backend](#connecting-frontend-to-backend)
4. [Environment Variables](#environment-variables)
5. [Troubleshooting](#troubleshooting)

## Frontend Deployment (Vercel)

### Prerequisites

- GitHub account with your repo
- Vercel account (free tier available)

### Step 1: Prepare for Deployment

```bash
cd Frontend

# Test build locally
npm run build

# Verify build succeeds
npm run preview
```

### Step 2: Connect to Vercel

1. Go to https://vercel.com
2. Click "New Project"
3. Connect your GitHub repository
4. Select the `GAN/Frontend` folder as root
5. Configure build settings:
   - **Framework**: Vite
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
   - **Install Command**: `npm install`

### Step 3: Set Environment Variables

In Vercel project settings:

1. Go to Settings → Environment Variables
2. Add:
   ```
   VITE_API_URL=https://your-backend-api.railway.app
   ```
   (Replace with your actual Railway backend URL)

### Step 4: Deploy

1. Click "Deploy"
2. Wait for build to complete
3. Your frontend is live at `https://your-project.vercel.app`

### Redeploy After Changes

```bash
# After pushing to GitHub
git push origin main

# Vercel automatically rebuilds and redeploys
```

## Backend Deployment (Railway)

### Prerequisites

- GitHub account with your repo
- Railway account (free tier available)
- NVIDIA GPU or sufficient resources

### Step 1: Prepare Backend for Railway

1. Update `Backend/requirements.txt` (already done)
2. Ensure no hardcoded localhost references
3. Set up environment handling for production

### Step 2: Create Railway Project

1. Go to https://railway.app
2. Click "New Project"
3. Select "Deploy from GitHub"
4. Select your repository

### Step 3: Configure Railway Project

1. **Root Directory**: Set to `GAN/Backend`
2. **Python Version**: 3.10 (should auto-detect)
3. **Port**: 8000 (FastAPI will use this)

### Step 4: Environment Variables

In Railway dashboard, set:

```
PYTHONUNBUFFERED=1
PORT=8000
```

### Step 5: Create Start Command

Railway should auto-detect from `main.py`, but verify:

```
python main.py --host 0.0.0.0 --port 8000
```

Or update `main.py` bottom section to use environment variables:

```python
if __name__ == "__main__":
    import os
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
    )
```

### Step 6: Deploy

1. Click "Deploy"
2. Wait for build
3. Once complete, Railway provides a public URL

## Connecting Frontend to Backend

### Update Environment Variables

After deploying backend to Railway:

1. Get your Railway backend URL from Railway dashboard
   - Format: `https://your-project-production.up.railway.app`

2. Update Vercel environment:
   - Go to Vercel project settings
   - Update `VITE_API_URL` to your Railway URL
   - Redeploy frontend

### Verify Connection

```bash
# Test from deployed frontend
curl https://your-railway-backend.up.railway.app/health

# Should return:
# {"status":"healthy","trainer_initialized":false}
```

## Environment Variables

### Frontend (Vercel)

| Variable | Value | Notes |
|----------|-------|-------|
| `VITE_API_URL` | `https://your-backend.up.railway.app` | Backend API URL |

### Backend (Railway)

| Variable | Value | Notes |
|----------|-------|-------|
| `PYTHONUNBUFFERED` | `1` | Real-time logging |
| `PORT` | `8000` | Container port |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU selection (if available) |

## Production Configuration Updates

### Backend `config.yaml` for Production

```yaml
training:
  epochs: 100
  batch_size: 64
  learning_rate: 0.0002

image:
  resolution: 128
  channels: 3

generator:
  latent_dim: 100
  feature_maps: 64

discriminator:
  feature_maps: 64

sampling:
  sample_interval: 100
  num_samples: 16

data:
  dataset_path: "/tmp/dcgan_data"
  train_split: 0.8
  num_workers: 2

output:
  samples_dir: "/tmp/dcgan_samples"
  models_dir: "/tmp/dcgan_models"
  logs_dir: "/tmp/dcgan_logs"

device:
  use_gpu: true
```

## Scaling & Optimization

### For Higher Performance

1. **Increase GPU Memory**: Railway's paid tier
2. **Larger Batch Size**: Adjust in config
3. **Higher Resolution**: Start with 128×128

### For Cost Efficiency

1. **Reduce Batch Size**: Train slower but save memory
2. **Lower Resolution**: Train at 64×64 first
3. **Fewer Epochs**: Start with 20-30 for testing

## Database/Storage (Optional)

If you want persistent storage across deployments:

### Using Railway Volumes

```bash
# In Railway dashboard:
1. Create a volume
2. Mount at /tmp/dcgan_storage
3. Update config to use this path
```

### Using External Storage

```bash
# Upload samples/models to AWS S3, Google Cloud Storage, etc.
# Add environment variables for credentials
```

## Monitoring & Logs

### Vercel

- Logs automatically available in Vercel dashboard
- View in "Deployments" → "Logs"

### Railway

- Logs available in Railway dashboard
- Click "Logs" in project view
- Real-time streaming with `PYTHONUNBUFFERED=1`

## Common Deployment Issues

### Issue: Backend Not Responding

**Solution**:
1. Check Railway logs for errors
2. Verify `VITE_API_URL` in Vercel matches Railway URL
3. Test with: `curl https://your-backend-url/health`

### Issue: CORS Errors

**Solution**: Already handled in FastAPI app with:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Issue: Training Data Not Found

**Solution**: 
- Upload dataset to Railway via Railway volumes
- Or use environment variables to point to cloud storage

### Issue: GPU Not Available

**Solution**:
- Railway free tier uses CPU
- Subscribe to Railway paid tier for GPU access
- Or deploy backend locally with GPU

## Custom Domain Setup

### Using Custom Domain with Vercel

1. Go to Vercel project settings
2. Add domain in "Domains"
3. Follow DNS instructions for your registrar

### Using Custom Domain with Railway

1. Railway provides auto-generated domain
2. For custom domain, use Railway paid tier
3. Or use Cloudflare as proxy

## Backup & Disaster Recovery

### Regular Backups

```bash
# Download trained models
curl https://your-backend/models | jq '.models'

# Download samples
curl https://your-backend/samples | jq '.samples'
```

### Restore from Backup

1. Store model files securely
2. Upload to Railway volumes when needed
3. Update `models_dir` config

## Security Considerations

### Current Setup (Development)

⚠️ The current setup allows public access to:
- Training endpoints
- Sample gallery
- Model listings

### Production Security Improvements

1. **Add Authentication**:
```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.get("/train/start")
async def start_training(credentials: HTTPAuthCredentials = Depends(security)):
    if credentials.credentials != os.getenv("API_KEY"):
        raise HTTPException(status_code=403)
    # ... rest of function
```

2. **Rate Limiting**:
```bash
pip install slowapi
```

3. **HTTPS Enforcement**:
   - Both Vercel and Railway provide automatic HTTPS

## Monitoring & Alerts

### Set Up Alerts

**Vercel**:
- Automatic notifications for build failures
- Email alerts for deployment issues

**Railway**:
- Set up metrics monitoring
- Get alerts for high resource usage

## Cost Estimation

| Service | Free Tier | Paid Tier |
|---------|-----------|-----------|
| Vercel | Unlimited deployments | $20+/month |
| Railway | 5GB storage + 100 hours compute | $5+/month |
| Total | ~$0 | ~$25/month |

## Next Steps

1. ✅ Deploy frontend to Vercel
2. ✅ Deploy backend to Railway
3. ✅ Connect them with environment variables
4. ✅ Test in production
5. ✅ Set up monitoring and backups
6. ✅ Share with team

---

**Successfully deployed? Great! Your DCGAN system is now live. 🚀**

For issues, check logs and deployment guides above.
