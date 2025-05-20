# Whisper Transcription API with YouTube Support

This API allows you to transcribe audio from various sources including direct audio file URLs and YouTube videos/playlists.

## Features

- Transcribe audio files directly from URLs
- Transcribe audio from YouTube videos
- Process entire YouTube playlists with a single request
- Automatic language detection
- Timestamps for each transcribed segment
- Automatic LLM summarization from deepseek

## Usage

The API accepts HTTP POST requests with a JSON payload containing the transcription parameters.

### Basic Request Format

```json
{
  "input": {
    "audio_file": "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav",
    "language": "en",
  }
}
```

### Input Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `audio_file` | string | Yes | - | URL to audio file or YouTube video/playlist |
| `language` | string | No | `null` | Language code (e.g., "en", "fr", "es"). If not provided, language will be auto-detected |
| `language_detection_min_prob` | float | No | 0 | Minimum probability threshold for language detection |
| `language_detection_max_tries` | int | No | 5 | Maximum number of attempts for language detection |
| `initial_prompt` | string | No | `null` | Text prompt to guide the transcription |
| `batch_size` | int | No | 64 | Batch size for processing |
| `temperature` | float | No | 0 | Temperature parameter for sampling |
| `vad_onset` | float | No | 0.500 | Voice activity detection onset threshold |
| `vad_offset` | float | No | 0.363 | Voice activity detection offset threshold |
| `align_output` | boolean | No | false | Whether to return word-level time-stamp transcriptions |
| `debug` | boolean | No | false | Enable debug mode |

## Example Requests

### Basic Example
```
{
  "input": {
    "audio_file": "https://www.youtube.com/watch?v=EsnULp2QYN0"
  }
}
```

### Transcribe a Direct Audio File

```json
{
  "input": {
    "audio_file": "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav",
    "language": "en"
  }
}
```

### Transcribe a YouTube Video

```json
{
  "input": {
    "audio_file": "https://www.youtube.com/watch?v=EsnULp2QYN0",
    "language": "en",
  }
}
```

### Transcribe a YouTube Playlist

```json
{
  "input": {
    "audio_file": "https://www.youtube.com/watch?v=EsnULp2QYN0&list=PLcMKRQM66creks1bUhOQqHuw0JaIfIztB",
    "language": "en",
  }
}
```

## Example Responses

### Single Audio File Response

```json
{
  "file_path": "gettysburg.wav",
  "segments": [
    {
      "text": " Four score and seven years ago, our fathers brought forth on this continent a new nation, conceived in liberty and dedicated to the proposition that all men are created equal.",
      "start": 0.009,
      "end": 10.009
    }
  ],
  "detected_language": "en"
}
```

### YouTube Playlist Response

```json
{
  "results": [
    {
      "file_path": "1-Video Title.mp3",
      "segments": [
        {
          "text": "This is the transcription of the first video.",
          "start": 0.5,
          "end": 5.2
        }
      ],
      "detected_language": "en"
    },
    {
      "file_path": "2-Another Video.mp3",
      "segments": [
        {
          "text": "This is the transcription of the second video.",
          "start": 0.2,
          "end": 4.8
        }
      ],
      "detected_language": "en"
    }
  ],
  "total_files": 2
}
```

### Error Response

```json
{
  "error": "Failed to download audio from YouTube URL"
}
```

## Notes

- When processing YouTube playlists, all videos in the playlist will be transcribed and returned as an array of results
- Audio files are cleaned up automatically after processing
- Language detection is performed if no language is specified

## Requirements

This API requires the following:
- yt-dlp for YouTube video downloading
- ffmpeg for audio processing
- PyTorch with CUDA support for GPU acceleration

## Limitations

- Too large of files might have issues with yt-dlp / downloading the audio.
- YouTube videos with age restrictions or other access limitations may not be processed
