# 🚀 How to Deploy to Vercel

The easiest way to deploy this Next.js app is using the [Vercel Platform](https://vercel.com/new).

## 📋 Prerequisites
1. Ensure your project is pushed to a Git repository (GitHub, GitLab, or Bitbucket).
2. Create a [Vercel Account](https://vercel.com/signup).

## 🛠️ Step-by-Step Guide

### 1. Import Project
- Go to your Vercel Dashboard.
- Click **"Add New..."** -> **"Project"**.
- Select your Git repository from the list.

### 2. Configure Project (Critical Step!)
Before clicking "Deploy", you **must** configure the **Root Directory** because your Next.js app lives inside the `view` folder.

- In the "Configure Project" screen, look for **"Root Directory"**.
- Click **"Edit"**.
- Select the **`view`** folder.
- Click **"Continue"**.

### 3. Environment Variables
- Expand the **"Environment Variables"** section.
- Add your Google Maps Key:
  - **Name**: `NEXT_PUBLIC_GOOGLE_MAPS_API_KEY`
  - **Value**: `Your_Google_Maps_Key_Here`
  *(If you skip this, the map will show a "Map Unavailable" placeholder, but the app will still work).*

### 4. Deploy
- Click **"Deploy"**.
- Wait for the build to finish (usually 1-2 minutes).
- Once done, you will get a live URL (e.g., `traffic-prediction.vercel.app`).

---

## 🏃‍♂️ Verification
- Visit your new live URL.
- Try **Sign In** using the demo credentials:
  - **Email**: `demo@example.com`
  - **Password**: `demo`
- Test the **Route Planner** to see the map (if you added the key).

**Enjoy your live app!** 🚀
