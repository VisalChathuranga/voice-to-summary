conda env create -f environment.yml

conda activate med-pipeline

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

curl -X POST "http://localhost:8000/api/transcribe-and-summarize" \
  -F "file=@/D:/MedCube/Projects/1st Month/aws-voice-to-text/audio/recording_2025-09-28T08-27-31-220Z.mp3"
