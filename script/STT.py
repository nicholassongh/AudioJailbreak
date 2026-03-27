import whisper
import json
import jsonlines

def transcribe_audio_files():
    # Load Whisper model
    model = whisper.load_model("base")

    # Read JSONL file
    with jsonlines.open('data.jsonl', 'r') as reader:
        # Read all data into a list for later updating
        data = list(reader)

    # Process each record
    for item in data:
        if 'response_audio_path' in item:
            # Transcribe audio file
            result = model.transcribe(item['response_audio_path'])
            # Add transcription text to data
            item['transcribed_text'] = result["text"]

    # Write updated data back to JSONL file
    with jsonlines.open('data.jsonl', 'w') as writer:
        writer.write_all(data)

if __name__ == "__main__":
    transcribe_audio_files()
