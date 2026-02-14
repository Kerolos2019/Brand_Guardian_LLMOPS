import os
import logging
import subprocess
import json
from pathlib import Path
import yt_dlp
from openai import OpenAI

logger = logging.getLogger("video-indexer")

class VideoIndexerService:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.temp_dir = Path("./temp_videos")
        self.temp_dir.mkdir(exist_ok=True)

    def download_youtube_video(self, url, output_path="temp_video.mp4"):
        """Downloads a YouTube video to a local file."""
        logger.info(f"Downloading YouTube video: {url}")

        ydl_opts = {
            'format': 'best',
            'outtmpl': output_path,
            'quiet': False,
            'no_warnings': False,
            'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            logger.info("Download complete.")
            return output_path
        except Exception as e:
            raise Exception(f"YouTube Download Failed: {str(e)}")

    def extract_audio(self, video_path):
        """Extracts audio from video file using ffmpeg."""
        audio_path = video_path.replace('.mp4', '.mp3')
        logger.info(f"Extracting audio from {video_path}...")

        try:
            # Use ffmpeg to extract audio
            subprocess.run([
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'libmp3lame',
                '-ar', '16000',  # 16kHz sample rate (optimal for Whisper)
                '-ac', '1',  # Mono
                '-b:a', '64k',  # Bitrate
                audio_path,
                '-y'  # Overwrite without asking
            ], check=True, capture_output=True)

            logger.info(f"Audio extracted to {audio_path}")
            return audio_path
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
            raise Exception(f"Audio extraction failed: {str(e)}")
        except FileNotFoundError:
            raise Exception("FFmpeg not found. Please install ffmpeg: https://ffmpeg.org/download.html")

    def transcribe_audio(self, audio_path):
        """Transcribes audio using OpenAI Whisper API."""
        logger.info(f"Transcribing audio with OpenAI Whisper...")

        try:
            with open(audio_path, 'rb') as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",  # Get detailed timestamps and metadata
                    language="en"  # Specify language or remove for auto-detection
                )

            logger.info("Transcription complete.")
            return transcript

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise Exception(f"OpenAI Whisper transcription failed: {str(e)}")

    def extract_text_from_video(self, video_path):
        """
        Extract on-screen text from video frames using OpenAI Vision API.
        Samples frames at regular intervals and extracts text.
        """
        logger.info(f"Extracting on-screen text from video...")
        ocr_texts = []

        try:
            # Get video duration using ffprobe
            result = subprocess.run([
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'json',
                video_path
            ], capture_output=True, text=True, check=True)

            duration = float(json.loads(result.stdout)['format']['duration'])
            logger.info(f"Video duration: {duration:.2f} seconds")

            # Sample frames every 10 seconds (adjust as needed)
            sample_interval = 10
            num_samples = min(int(duration / sample_interval), 6)  # Max 6 samples to avoid too many API calls

            frame_dir = self.temp_dir / "frames"
            frame_dir.mkdir(exist_ok=True)

            for i in range(num_samples):
                timestamp = i * sample_interval
                frame_path = frame_dir / f"frame_{i}.jpg"

                # Extract frame at timestamp
                subprocess.run([
                    'ffmpeg',
                    '-ss', str(timestamp),
                    '-i', video_path,
                    '-frames:v', '1',
                    '-q:v', '2',  # High quality
                    str(frame_path),
                    '-y'
                ], check=True, capture_output=True)

                # Use OpenAI Vision API to extract text from frame
                try:
                    import base64
                    with open(frame_path, 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode('utf-8')

                    response = self.openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Extract ALL visible text from this video frame. Return only the text you see, nothing else. If no text is visible, respond with 'NO_TEXT'."
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{img_data}"
                                        }
                                    }
                                ]
                            }
                        ],
                        max_tokens=500
                    )

                    extracted_text = response.choices[0].message.content.strip()
                    if extracted_text and extracted_text != "NO_TEXT":
                        ocr_texts.append(extracted_text)
                        logger.info(f"Frame {i} text: {extracted_text[:50]}...")

                except Exception as e:
                    logger.warning(f"Failed to extract text from frame {i}: {e}")

            # Cleanup frames
            for frame_file in frame_dir.glob("*.jpg"):
                frame_file.unlink()

            return ocr_texts

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return []  # Return empty list if OCR fails (non-critical)

    def process_video(self, video_path, video_name):
        """
        Complete video processing pipeline:
        1. Extract audio
        2. Transcribe with Whisper
        3. Extract on-screen text with Vision API
        4. Get video metadata
        """
        logger.info(f"Processing video: {video_name}")

        # Extract and transcribe audio
        audio_path = self.extract_audio(video_path)
        transcript_data = self.transcribe_audio(audio_path)

        # Extract on-screen text (OCR)
        ocr_text = self.extract_text_from_video(video_path)

        # Get video duration
        result = subprocess.run([
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            video_path
        ], capture_output=True, text=True, check=True)
        duration = float(json.loads(result.stdout)['format']['duration'])

        # Cleanup temporary files
        if os.path.exists(audio_path):
            os.remove(audio_path)

        return {
            "transcript": transcript_data.text,
            "ocr_text": ocr_text,
            "video_metadata": {
                "duration": duration,
                "platform": "youtube",
                "language": transcript_data.language if hasattr(transcript_data, 'language') else "unknown"
            }
        }