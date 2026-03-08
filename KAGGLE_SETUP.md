# Kaggle API Setup (Step 4.1)

## Get `kaggle.json`

1. Go to **https://www.kaggle.com** → sign in.
2. Click your **profile picture** (top right) → **Settings**.
3. Scroll to **API** → click **Create New Token**.
4. This downloads **`kaggle.json`** (usually to your **Downloads** folder).

## Install it (Mac/Linux)

The Kaggle CLI looks for credentials in `~/.kaggle/kaggle.json`. Do this **from the folder where `kaggle.json` is** (e.g. Downloads), or use the full path.

### Option A — From the folder where `kaggle.json` is

```bash
# If it's in Downloads:
cd ~/Downloads
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Option B — Using full path (any directory)

```bash
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

`chmod 600` makes the file readable only by you, which Kaggle requires.

## Check it works

```bash
kaggle datasets list
```

If you see a list of datasets, the API is connected.

## Next: download the Spotify dataset

```bash
kaggle datasets download -d maharshipandya/-spotify-tracks-dataset
unzip -o -d data/raw ./-spotify-tracks-dataset.zip
```

(Or the exact dataset URL from your project — adjust the dataset path if different.)
