import os
from youtube_transcript_api import YouTubeTranscriptApi

def get_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text_data = " ".join([entry['text'] for entry in transcript])
    
    save_dir = "transcript"
    
    # Create save directory if it doesn't exist
    save_path = os.path.join(os.path.dirname(__file__), save_dir)
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Save Path: {save_path}\n")
    
    # Save transcript to file
    file_path = os.path.join(save_path, f"{video_id}.txt")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text_data)
    
    print(f"Transcript saved at: {file_path}")
    
    return text_data

if __name__ == "__main__":
    video_id = 'cVRuLaEJij0&t=890s'
    print(get_transcript(video_id))