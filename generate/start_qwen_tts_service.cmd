@echo off
E:\Code\faster-qwen3-tts\.venv\Scripts\python.exe E:\Code\faster-qwen3-tts\generate\qwen_http_service.py ^
  --script E:\Code\faster-qwen3-tts\generate\generate_mp3_with_embedding.py ^
  --config E:\Code\faster-qwen3-tts\generate\qwen_http_service_config.json ^
  --voices-dir E:\Code\faster-qwen3-tts\voices ^
  --extract-script E:\Code\faster-qwen3-tts\examples\extract_speaker.py ^
  --voice-model Qwen/Qwen3-TTS-12Hz-1.7B-Base ^
  --host 0.0.0.0 --port 8765
