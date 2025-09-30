# Voice → Roles → Summary (FastAPI + AWS Transcribe)

Convert an uploaded **.mp3 / .webm** into a **role‑labeled conversation** (Doctor/Patient/Nurse/Other) and return a **concise medical summary**.  
Also writes a single human‑friendly transcript file `*_conversation.txt` on the server and exposes a **download URL** so a client app can save it locally.

---

## ✨ Features
- Upload **.mp3 / .webm** → AWS Transcribe (Medical or Standard)
- **Speaker role classification** (Doctor / Patient / Nurse / Other)
- **Clinical summary** (keeps your Project 2 prompt)
- **Single transcript file** per job, e.g.:  
  `recording_20250928_082731_a1b2c3_conversation.txt`
- Optimized for speed: S3 Transfer Acceleration, multipart uploads, short polling
- Clean **JSON response**: `{ "summary_text": "...", "download_url": "/api/download/..." }`

---

## 🧰 Prerequisites

- **Conda** (or mamba)
- **Python 3.10+**
- **FFmpeg** on PATH (required by pydub)
  - Windows: `winget install Gyan.FFmpeg` or `choco install ffmpeg`  
  - macOS: `brew install ffmpeg`  
  - Linux: `sudo apt-get install ffmpeg`

- AWS credentials with permissions for:
  - S3: create bucket (if needed), put/get objects
  - Transcribe (and Transcribe Medical if `USE_MEDICAL=true`)

---

## 🚀 Setup

```bash
# 1) Create env
conda env create -f environment.yml

# 2) Activate
conda activate med-pipeline
```

Create a **.env** in the project root:

```ini
# ---- AWS / Transcribe ----
AWS_ACCESS_KEY_ID=YOUR_KEY
AWS_SECRET_ACCESS_KEY=YOUR_SECRET
REGION=us-east-1
BUCKET=my-voice2text-us-east1

# ---- Transcribe flavor ----
USE_MEDICAL=true
SPECIALTY=primarycare
LANGUAGE=en-US

# ---- S3 transfer tuning ---- (optional; enabled by default)
S3_ACCELERATE=true
S3_ENABLE_ACCELERATE_IF_NEEDED=true
S3_MAX_CONCURRENCY=16
S3_MULTIPART_THRESHOLD_MB=8
S3_MULTIPART_CHUNKSIZE_MB=8

# ---- Audio re-encode ----
FORCE_REENCODE=true
TARGET_SAMPLE_RATE=16000
TARGET_CHANNELS=1
TARGET_BITRATE=64k

# ---- OpenAI (role classification + summary) ----
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# ---- Output dirs ----
TRANSCRIPTS_DIR=transcripts  # server saves the single *_conversation.txt here
```

> The app will create the `transcripts/` folder automatically on first run.

---

## 🏃‍♂️ Run the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

You should see logs that include S3 acceleration info and upload mode.

---

## 🧪 Test with cURL

```bash
curl -X POST "http://localhost:8000/api/transcribe-and-summarize"   -H "Accept: application/json"   -F "file=@/D:/MedCube/Projects/1st Month/aws-voice-to-text/audio/recording_2025-09-28T08-27-31-220Z.mp3"
```

**Response**
```json
{
  "summary_text": "A patient presents with ...",
  "download_url": "/api/download/recording_20250928_082731_a1b2c3_conversation.txt"
}
```

> The `.txt` is stored on the **server** in `transcripts/`.  
> Clients can download it via the `download_url` endpoint (see C# client below).

---

## 🔌 API

### `POST /api/transcribe-and-summarize`
- **Body** (multipart/form-data): `file` = .mp3 or .webm
- **Returns**:  
  ```json
  { "summary_text": "…", "download_url": "/api/download/<file>.txt" }
  ```

### `GET /api/download/{name}`
- Downloads the transcript file by name (served from `TRANSCRIPTS_DIR`).

---

## 🖥️ C#/.NET 8 Client (Console)

This .NET console app:
1) Uploads a local `.mp3` / `.webm` to your FastAPI server
2) Prints the summary
3) Downloads the transcript to the **client’s Downloads folder**

### Create project

```bash
dotnet new console -n VoiceSummaryClient
cd VoiceSummaryClient
```

### Replace `Program.cs` with:

```csharp
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text.Json;
using System.Runtime.InteropServices;

class Program
{
    static async Task Main(string[] args)
    {
        // ---- CONFIG ----
        var apiBase = "http://YOUR_SERVER_HOST:8000"; // e.g. http://localhost:8000 or http://<server-ip>:8000
        var audioPath = args.Length > 0 ? args[0] : @"D:\path\to\audio.webm"; // or .mp3

        if (!File.Exists(audioPath))
        {
            Console.WriteLine($"File not found: {audioPath}");
            return;
        }

        using var http = new HttpClient { BaseAddress = new Uri(apiBase) };

        Console.WriteLine("Uploading & processing…");

        using var form = new MultipartFormDataContent();
        var fileContent = new StreamContent(File.OpenRead(audioPath));
        fileContent.Headers.ContentType = new MediaTypeHeaderValue(GetMimeType(audioPath));
        form.Add(fileContent, "file", Path.GetFileName(audioPath));

        var resp = await http.PostAsync("/api/transcribe-and-summarize", form);
        var body = await resp.Content.ReadAsStringAsync();

        if (!resp.IsSuccessStatusCode)
        {
            Console.WriteLine($"HTTP {(int)resp.StatusCode}: {body}");
            return;
        }

        var json = JsonSerializer.Deserialize<ApiResponse>(body, new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        });

        Console.WriteLine("\n--- SUMMARY ---\n");
        Console.WriteLine(json?.SummaryText ?? "(empty)");

        if (string.IsNullOrWhiteSpace(json?.DownloadUrl))
        {
            Console.WriteLine("\n(No download_url returned by server — skipping file download.)");
            return;
        }

        // Resolve absolute URL if server returned a relative path
        var downloadUri = json!.DownloadUrl.StartsWith("http", StringComparison.OrdinalIgnoreCase)
            ? new Uri(json.DownloadUrl)
            : new Uri(http.BaseAddress!, json.DownloadUrl);

        // Decide local Downloads folder
        var downloads = GetDownloadsFolder();
        Directory.CreateDirectory(downloads);

        // Infer filename from URL
        var fileName = Path.GetFileName(downloadUri.LocalPath);
        var savePath = Path.Combine(downloads, fileName);

        Console.WriteLine($"\nDownloading transcript to: {savePath}");
        var fileResp = await http.GetAsync(downloadUri);
        fileResp.EnsureSuccessStatusCode();
        await using (var fs = File.Create(savePath))
        {
            await fileResp.Content.CopyToAsync(fs);
        }
        Console.WriteLine("Saved.");
    }

    static string GetMimeType(string path)
    {
        var ext = Path.GetExtension(path).ToLowerInvariant();
        return ext switch
        {
            ".mp3"  => "audio/mpeg",
            ".webm" => "audio/webm",
            _       => "application/octet-stream"
        };
    }

    static string GetDownloadsFolder()
    {
        // Windows: %USERPROFILE%\Downloads
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            var user = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            return Path.Combine(user, "Downloads");
        }
        // macOS/Linux: ~/Downloads
        var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        return Path.Combine(home, "Downloads");
    }

    class ApiResponse
    {
        public string? SummaryText { get; set; }
        public string? DownloadUrl { get; set; }
    }
}
```

### Build & Run

```bash
dotnet build
dotnet run -- "D:\MedCube\Projects\...\recording_2025-09-28T08-27-31-220Z.mp3"
```

You’ll see the summary in the console and the transcript `.txt` saved in your **Downloads**.

> If your server is remote, make sure the client machine can reach `http://YOUR_SERVER_HOST:8000/api/download/...`

---

## ⚙️ Troubleshooting

- **It’s slow**: Most time is AWS Transcribe compute. We already poll every 3s and use S3 acceleration.
- **S3 acceleration not enabled**: Check server logs; you should see messages like:
  - `S3 Transfer Acceleration already enabled for bucket: ...`
- **FFmpeg not found**: `pydub` requires ffmpeg. Install it and ensure it’s on PATH.
- **CORS**: If your client is on a different origin, enable CORS in `app.main`:
  ```python
  from fastapi.middleware.cors import CORSMiddleware
  app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],  # or specific origins
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )
  ```

---

## 📄 License

MIT (or your preferred license)
