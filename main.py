import os
import re
import sys
import argparse
import json
import torch
import yt_dlp
import whisperx
import anthropic
from typing import Dict, Any
from datetime import datetime
from urllib.parse import parse_qs, urlparse


CLAUDE_API_KEY_FILE_PATH = "resources/claude_api_key.txt"
PROMPT_TEMPLATE_FILE_PATH = "resources/prompt.txt"

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='YouTube video download, transcription and Claude API processing')
    parser.add_argument('--url', type=str, required=True, help='YouTube video URL')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--whisper_model', type=str, default='turbo', help='Whisper model size')
    parser.add_argument('--force', action='store_true', help='Force reprocessing of already processed steps')
    parser.add_argument('--target_language', type=str, default='Korean', help='Target language for vocabulary (e.g., Korean, Japanese, Chinese)')
    return parser.parse_args()

def get_video_id(url: str) -> str:
    """Extract video ID from YouTube URL"""
    parsed_url = urlparse(url)
    if parsed_url.netloc == 'youtu.be':
        return parsed_url.path[1:]
    if parsed_url.netloc in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query)['v'][0]
        if parsed_url.path.startswith('/embed/'):
            return parsed_url.path.split('/')[2]
        if parsed_url.path.startswith('/v/'):
            return parsed_url.path.split('/')[2]
    # Unsupported URL format
    raise ValueError(f'Unsupported YouTube URL format: {url}')

def setup_video_directories(video_id: str, base_output_dir: str) -> Dict[str, str]:
    """Set up directories for each video ID"""
    video_dir = os.path.join(base_output_dir, video_id)
    audio_dir = os.path.join(video_dir, "1_audio")
    transcript_dir = os.path.join(video_dir, "2_transcript")
    vocabulary_dir = os.path.join(video_dir, "3_vocabulary")
    
    # Create directories
    for directory in [audio_dir, transcript_dir, vocabulary_dir]:
        os.makedirs(directory, exist_ok=True)
    
    return {
        "video_dir": video_dir,
        "audio_dir": audio_dir,
        "transcript_dir": transcript_dir,
        "vocabulary_dir": vocabulary_dir
    }

def download_youtube_audio(url: str, audio_dir: str, force: bool = False) -> str:
    """Download audio from YouTube (skip if already exists)"""
    print(f"Processing YouTube video: {url}")
    
    # Check if audio file already exists
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    if audio_files and not force:
        audio_file = os.path.join(audio_dir, audio_files[0])
        print(f"Found existing audio file: {audio_file}")
        return audio_file
    
    # yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(audio_dir, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': False,
        'no_warnings': False
    }
    
    # Execute download
    print(f"Starting audio download...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_filename = ydl.prepare_filename(info).replace(f".{info['ext']}", ".wav")
    
    print(f"Download complete: {audio_filename}")
    return audio_filename

def transcribe_with_whisperx(audio_file: str, transcript_dir: str, model_name: str = 'large-v2', force: bool = False) -> str:
    """Transcribe audio using WhisperX"""
    # Check if transcription file already exists
    transcript_file = os.path.join(transcript_dir, "transcription.txt")
    if os.path.exists(transcript_file) and not force:
        print(f"Found existing transcription file: {transcript_file}")
        with open(transcript_file, 'r', encoding='utf-8') as f:
            full_text = f.read()
        return full_text
    
    print(f"Transcribing with WhisperX: {audio_file}")
    
    # Force CPU usage
    device = "cpu"
    print(f"WhisperX running on: {device}")
    
    # Load WhisperX model - use float32 on CPU
    model = whisperx.load_model(model_name, device, compute_type="float32")
    
    # Load audio and transcribe
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=16)
    
    # Detect language
    detected_language = result["language"]
    print(f"Detected language: {detected_language}")
    
    # Extract pure text
    full_text = " ".join([segment["text"] for segment in result["segments"]])
    
    # Save text file
    with open(transcript_file, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    print(f"Transcription complete. Result saved to: {transcript_file}")
    return full_text

def query_claude_api(text: str, prompt_template_path: str, api_key_path: str, vocabulary_dir: str, target_language: str = "Korean", force: bool = False) -> str:
    """Analyze text using Claude API"""
    # Check if vocabulary file already exists
    vocabulary_file = os.path.join(vocabulary_dir, "vocabulary.md")
    if os.path.exists(vocabulary_file) and not force:
        print(f"Found existing vocabulary file: {vocabulary_file}")
        with open(vocabulary_file, 'r', encoding='utf-8') as f:
            response = f.read()
        return response
    
    print("Calling Claude API...")
    
    with open(api_key_path, 'r') as f:
        api_key = f.readline().strip()
        
    with open(prompt_template_path, 'r') as f:
        prompt_template = f.read().strip()
    
    # Replace target language in the prompt template
    prompt = prompt_template.replace("TARGET_LANGUAGE", target_language)
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # Construct final prompt
    full_prompt = f"{prompt}\n\nText:\n{text}"
    
    try:
        response = client.messages.create(
            model="claude-3-7-sonnet-latest",  # Use latest model
            max_tokens=4000,
            messages=[
                {"role": "user", "content": full_prompt}
            ]
        )
        claude_response = response.content[0].text
        
        # Save raw response (for debugging)
        raw_response_file = os.path.join(vocabulary_dir, "raw_response.txt")
        with open(raw_response_file, 'w', encoding='utf-8') as f:
            f.write(claude_response)
            
        return claude_response
    except Exception as e:
        print(f"Error occurred while calling Claude API: {e}")
        return f"Error: {str(e)}"

def save_claude_response(response: str, vocabulary_dir: str, audio_file: str = None) -> str:
    """Save Claude API response in Markdown card format"""
    # Generate current date for filename
    current_date = datetime.now().strftime("%Y%m%d")
    
    # Extract original video title from audio filename
    if audio_file:
        # Get just the filename without extension
        video_title = os.path.splitext(os.path.basename(audio_file))[0]
    else:
        # Fallback to video ID if audio_file is not provided
        video_title = os.path.basename(os.path.dirname(os.path.dirname(vocabulary_dir)))
    
    # Create a filename-safe version of the title (remove special characters)
    safe_title = re.sub(r'[^\w\s-]', '', video_title).strip().replace(' ', '_')
    
    vocabulary_file = os.path.join(vocabulary_dir, f"{current_date}_{safe_title}.md")

    # Extract word blocks using regex
    entries = re.split(r'-{3,}', response.strip())

    # Convert to card format
    formatted_entries = []
    
    # Try to find numbered expressions
    numbered_pattern = re.compile(r'^[\s*]*(\d+)[\s*]*\.\s*\[(.+?)\]:', re.MULTILINE)
    
    for entry in entries:
        if not entry.strip():
            continue  # Skip empty entries
            
        # Check if this entry has a numbered expression
        numbered_match = numbered_pattern.search(entry)
        
        if numbered_match:
            # Handle numbered expressions
            entry_number = numbered_match.group(1)
            word = numbered_match.group(2).strip()
            
            # Extract meaning, example and translation
            meaning_match = re.search(r'meaning \[(.+?)\]', entry, re.IGNORECASE)
            example_match = re.search(r'(?:example|예시): (.+)', entry, re.IGNORECASE)
            translation_match = re.search(r'(?:translation|해석): (.+)', entry, re.IGNORECASE)
            
            if meaning_match and example_match and translation_match:
                meaning = meaning_match.group(1).strip()
                example = example_match.group(1).strip()
                translation = translation_match.group(1).strip()
                
                card = f"""### {entry_number}. 📝 **{word}**  
- **Meaning**: {meaning}  
- **Example**: *{example}*  
- **Translation**: {translation}

---"""
                formatted_entries.append(card)
        else:
            # Legacy pattern for non-numbered expressions
            match_word = re.search(r'\[(.+?)\]: (?:뜻|meaning) \[(.+?)\]', entry, re.IGNORECASE)
            match_example = re.search(r'(?:예시|example): (.+)', entry, re.IGNORECASE)
            match_translation = re.search(r'(?:해석|translation): (.+)', entry, re.IGNORECASE)

            if match_word and match_example and match_translation:
                word = match_word.group(1).strip()
                meaning = match_word.group(2).strip()
                example = match_example.group(1).strip()
                translation = match_translation.group(1).strip()

                card = f"""### 📝 **{word}**  
- **Meaning**: {meaning}  
- **Example**: *{example}*  
- **Translation**: {translation}

---"""
                formatted_entries.append(card)

    # Use video title as the heading
    markdown_text = f"# {video_title}\n\n" + "\n\n".join(formatted_entries)

    with open(vocabulary_file, 'w', encoding='utf-8') as f:
        f.write(markdown_text)

    print(f"Claude response saved in Markdown card format: {vocabulary_file}")
    return vocabulary_file

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Extract video ID and set up directories
        video_id = get_video_id(args.url)
        directories = setup_video_directories(video_id, args.output_dir)
        
        print(f"Starting processing: Video ID {video_id}")
        print(f"Working directory: {directories['video_dir']}")
        
        # 1. Download YouTube video
        audio_file = download_youtube_audio(
            args.url, 
            directories['audio_dir'], 
            force=args.force
        )
        
        # 2. Transcribe with WhisperX
        transcript_text = transcribe_with_whisperx(
            audio_file, 
            directories['transcript_dir'], 
            args.whisper_model, 
            force=args.force
        )
        
        # 3. Query Claude API
        claude_response = query_claude_api(
            transcript_text, 
            PROMPT_TEMPLATE_FILE_PATH, 
            CLAUDE_API_KEY_FILE_PATH, 
            directories['vocabulary_dir'],
            args.target_language,
            force=args.force
        )
        
        # 4. Save response - pass audio_file to use its name
        vocabulary_file = save_claude_response(
            claude_response, 
            directories['vocabulary_dir'],
            audio_file
        )
        
        print(f"\nProcessing complete!")
        print(f"Video ID: {video_id}")
        print(f"Audio file: {audio_file}")
        print(f"Transcript file: {os.path.join(directories['transcript_dir'], 'transcription.txt')}")
        print(f"Vocabulary file: {vocabulary_file}")
        
    except Exception as e:
        print(f"Error occurred during processing: {e}")
        sys.exit(1)
        
        
if __name__ == "__main__":
    main()