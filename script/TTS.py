#text to speech
from google.cloud import texttospeech
import os
from dotenv import load_dotenv
import random
import logging
from datetime import datetime

class TTSService:
    def __init__(self, audio_save_path=None, audio_prefix=''):
        load_dotenv()
        self.client = texttospeech.TextToSpeechClient.from_service_account_file(
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        )
        self.voice_cache = {}
        self.initialize_voice_cache()
        self.audio_save_path = audio_save_path or os.path.join(os.getcwd(), 'audio_outputs')
        self.audio_prefix = audio_prefix
        # Ensure audio save directory exists
        os.makedirs(self.audio_save_path, exist_ok=True)

    def initialize_voice_cache(self):
        """Initialize voice configuration cache."""
        voices = self.client.list_voices().voices
        for voice in voices:
            for language_code in voice.language_codes:
                if language_code not in self.voice_cache:
                    self.voice_cache[language_code] = []
                self.voice_cache[language_code].append({
                    'name': voice.name,
                    'gender': voice.ssml_gender
                })

    def get_random_voice(self, language_code):
        """Get a random voice configuration for the specified language from cache."""
        if language_code not in self.voice_cache:
            raise ValueError(f"No voices found for language {language_code}")

        return random.choice(self.voice_cache[language_code])

    def text_to_speech(self, text, output_file, language='en-US', voice_name=None, gender=None):
        """
        Convert text to speech.

        :param text: Text to convert
        :param output_file: Output audio file path
        :param language: Language code, defaults to English
        :param voice_name: Optional specific voice name
        :param gender: Optional specific voice gender
        """
        if voice_name is None or gender is None:
            random_voice = self.get_random_voice(language)
            voice_name = voice_name or random_voice['name']
            gender = gender or random_voice['gender']

        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=language,
            name=voice_name,
            ssml_gender=gender
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = self.client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        with open(output_file, "wb") as out:
            out.write(response.audio_content)
            print(f'Audio content written to file "{output_file}"')

    def batch_process_jsonl(self, input_file, output_file):
        """
        Batch-process prompts from a JSONL file and generate speech.

        :param input_file: Input JSONL file path
        :param output_file: Output JSONL file path
        """
        import json
        import os

        # Set up logging
        log_dir = os.path.join(os.path.dirname(output_file), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'tts_errors_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        logging.basicConfig(
            filename=log_file,
            level=logging.ERROR,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # Use the instance's audio save path
        output_dir = self.audio_save_path
        os.makedirs(output_dir, exist_ok=True)

        processed_records = []
        checkpoint_file = output_file + '.checkpoint'

        # Check for checkpoint file
        last_processed_index = -1
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    last_processed_index = int(f.read().strip())
                print(f"Resuming from checkpoint, last processed index: {last_processed_index}")
            except Exception as e:
                logging.error(f"Failed to read checkpoint file: {str(e)}")

        try:
            # Read and process JSONL file
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())

                        # Skip if already processed
                        if record['index'] <= last_processed_index:
                            processed_records.append(record)
                            continue

                        # Generate audio filename
                        audio_filename = f"{self.audio_prefix}_prompt_{record['index']}.mp3"
                        audio_path = os.path.join(output_dir, audio_filename)

                        # Convert text to speech
                        if record['prompt']:
                            self.text_to_speech(
                                text=record['prompt'],
                                output_file=audio_path,
                                language="en-US"
                            )
                            record['speech_path'] = audio_path

                        processed_records.append(record)

                        # Update checkpoint
                        with open(checkpoint_file, 'w') as f:
                            f.write(str(record['index']))

                    except Exception as e:
                        error_msg = f"Error processing record {record.get('index', 'unknown')}: {str(e)}"
                        logging.error(error_msg)
                        print(error_msg)
                        continue

            # Write new JSONL file
            with open(output_file, 'w', encoding='utf-8') as f:
                for record in processed_records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')

            # Delete checkpoint file after completion
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)

        except Exception as e:
            error_msg = f"Error during batch processing: {str(e)}"
            logging.error(error_msg)
            print(error_msg)
            raise

# Example usage
if __name__ == "__main__":
    tts_service = TTSService()

    # Test Chinese
    tts_service.text_to_speech(
        text="你好，世界！",
        output_file="chinese_output.mp3",
        language="cmn-CN"
    )

    # Test English
    tts_service.text_to_speech(
        text="Hello, World!",
        output_file="english_output.mp3",
        language="en-US"
    )

    # Batch processing example
    tts_service.batch_process_jsonl(
        input_file="path/to/input.jsonl",
        output_file="path/to/output.jsonl"
    )
