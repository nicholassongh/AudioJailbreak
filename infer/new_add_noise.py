import numpy as np
from pydub import AudioSegment
import librosa
import os

class NoiseAdder:
    def __init__(self):
        # Initialize statistics counters
        self.noise_stats = {
            'pitch_shift': 0,     # Pitch adjustment
            'time_stretch': 0,    # Speed adjustment
            'frequency_noise': 0, # High/low frequency perturbation
            'dropout': 0,         # Frame dropout
            'rain_noise': 0,      # Rain sound
            'fade': 0,            # Fade effect
            'baby_cry': 0,        # Baby crying sound
            'car_horn': 0,        # Car horn
            'music': 0,           # Background music
            'volume': 0           # Volume adjustment
        }

        # Track detailed parameter usage per noise type
        self.detailed_stats = {
            'pitch_shift': {'up': 0, 'down': 0},
            'time_stretch': {'faster': 0, 'slower': 0},
            'frequency_noise': {'high': 0, 'low': 0},
            'rain_noise': {'heavy': 0, 'light': 0},
            'baby_cry': {'continuous': 0, 'intermittent': 0},
            'car_horn': {'single': 0, 'multiple': 0},
            'music': {'low_freq': 0, 'mid_freq': 0, 'high_freq': 0},
            'volume': {'up': 0, 'down': 0}
        }

        # Load noise files
        self._load_noise_files()

    def _load_noise_files(self):
        """Pre-load noise files for efficiency."""
        # Initialize background audio
        try:
            self.rain = AudioSegment.from_mp3("./noise_test/noise/rain.mp3") - 7
            self.baby_laugh = AudioSegment.from_mp3("./noise_test/noise/baby_laugh.mp3") - 7
            self.baby_cry = AudioSegment.from_mp3("./noise_test/noise/baby_cry.mp3") - 9
            self.car = AudioSegment.from_mp3("./noise_test/noise/car.mp3") - 7
            self.music = AudioSegment.from_mp3("./noise_test/noise/music.mp3") - 10
            self.noise_files_loaded = True
        except Exception as e:
            print(f"Warning: Could not load noise files: {e}")
            self.noise_files_loaded = False

    def get_stats(self):
        """Return noise usage statistics."""
        return {
            'noise_counts': self.noise_stats,
            'detailed_stats': self.detailed_stats,
            'total_noise_added': sum(self.noise_stats.values())
        }

    def print_stats(self):
        """Print noise usage statistics."""
        stats = self.get_stats()
        print("\nNoise addition statistics:")
        print(f"Total noise additions: {stats['total_noise_added']}")
        print("\nUsage count per noise type:")
        for noise_type, count in stats['noise_counts'].items():
            print(f"- {noise_type}: {count}")

        print("\nDetailed noise parameter statistics:")
        for noise_type, params in stats['detailed_stats'].items():
            if sum(params.values()) > 0:
                print(f"- {noise_type}: ", end="")
                details = [f"{param}={count}" for param, count in params.items() if count > 0]
                print(", ".join(details))

    def _validate_audio_file(self, file_path):
        """Validate audio file format and integrity."""
        if not os.path.exists(file_path):
            print(f"Error: File not found - {file_path}")
            return False

        try:
            audio = AudioSegment.from_file(file_path)
            # Check if audio duration exceeds 5 minutes
            if len(audio) > 5 * 60 * 1000:  # 5 minutes = 5 * 60 * 1000 ms
                print("Error: Audio length exceeds 5 minutes")
                return False
            # Check if audio is empty or corrupted
            if len(audio) == 0:
                print("Error: Audio file is empty")
                return False
            return True
        except Exception as e:
            print(f"Error: Invalid audio file format or file is corrupted - {e}")
            return False

    def add_noise(self, input_path, way=None, parameter=None):
        """
        Add noise to an audio file.

        Args:
            input_path: Path to the audio file
            way: Noise type parameter (0-1), randomly generated if None
            parameter: Noise parameter (0-1), randomly generated if None

        Returns:
            tuple: (way, parameter) on success, (None, None) if validation fails
        """
        # Validate input file
        if not self._validate_audio_file(input_path):
            return None, None

        # Initialize audio
        try:
            audio = AudioSegment.from_file(input_path)
        except Exception as e:
            print(f"Error: Could not load audio file - {e}")
            return None, None

        try:
            other_audio, sr = librosa.load(input_path, mono=True)  # mono ensures single channel
        except Exception as e:
            print(f"Error: librosa could not load audio - {e}")
            return None, None

        # Use provided way and parameter if given, otherwise generate randomly
        if way is not None and parameter is not None:
            random_float = round(way, 3)
            random_float2 = round(parameter, 3)
        else:
            # Original random generation
            random_float = round(np.random.uniform(0.0, 1.0), 3)
            random_float2 = round(np.random.uniform(0.0, 1.0), 3)

        print(f"2.add_noise random1: {random_float}, random2: {random_float2}")

        # Pitch shift
        if 0 < random_float < 0.1:
            self.noise_stats['pitch_shift'] += 1
            if random_float2 >= 0 and random_float2 < 0.5 and self.detailed_stats['pitch_shift']['up']<3:
                other_audio = librosa.effects.pitch_shift(other_audio, sr=sr, n_steps=random_float2 * 1)
                self.detailed_stats['pitch_shift']['up'] += 1
            elif self.detailed_stats['pitch_shift']['down']<4:
                other_audio = librosa.effects.pitch_shift(other_audio, sr=sr, n_steps=-random_float2 * 1)
                self.detailed_stats['pitch_shift']['down'] += 1

        # Time stretch
        elif 0.1 <= random_float < 0.2:
            self.noise_stats['time_stretch'] += 1
            # Ensure rate parameter is positive
            if random_float2 > 0 and random_float2 < 0.5 and self.detailed_stats['time_stretch']['faster']<3:
                rate = max(1.1, 1+random_float2 * 0.1)  # Ensure maximum is 1.1
                other_audio = librosa.effects.time_stretch(other_audio, rate=rate)
                self.detailed_stats['time_stretch']['faster'] += 1
            elif self.detailed_stats['time_stretch']['slower']<6:
                rate = max(0.9, 1-random_float2 * 0.1)  # Ensure minimum is 0.1
                other_audio = librosa.effects.time_stretch(other_audio, rate=rate)
                self.detailed_stats['time_stretch']['slower'] += 1

        # High/low frequency perturbation
        elif 0.2 <= random_float < 0.3 and self.noise_stats['frequency_noise']<3:
            self.noise_stats['frequency_noise'] += 1
            duration = len(other_audio) / sr  # Audio duration
            t = np.linspace(0, duration, len(other_audio), endpoint=False)
            if random_float2 > 0.5:
                # High-frequency perturbation (20000Hz)
                tone = 0.1 * np.sin(2 * np.pi * 20000*(1+random_float2) * t)
                self.detailed_stats['frequency_noise']['high'] += 1
            else:
                # Low-frequency perturbation (20Hz)
                tone = 0.1 * np.sin(2 * np.pi * 20*(1-random_float2) * t)
                self.detailed_stats['frequency_noise']['low'] += 1

            # Add perturbation to audio
            other_audio = other_audio + tone

        # Frame dropout
        elif 0.3 <= random_float < 0.4 and self.noise_stats['dropout']<3:
            self.noise_stats['dropout'] += 1
            begin_time = random_float2 * len(audio)
            end_time = begin_time + random_float2 * 100
            left_audio = audio[:begin_time]
            right_audio = audio[end_time:]
            audio = left_audio + right_audio

        # Add rain sound
        elif 0.4 <= random_float < 0.5 and self.noise_stats['rain_noise']<1:
            self.noise_stats['rain_noise'] += 1
            rain = self.rain[:len(audio)]
            rain_volume = random_float2 * 0.9  # Rain volume varies 0-90%
            rain_adjusted = rain - (8 * (1 - rain_volume))  # Adjust rain volume
            # Randomly decide heavy or light rain
            if random_float2 > 0.7:
                # Heavy rain - lower frequencies
                rain_adjusted = rain_adjusted.low_pass_filter(2000)
                self.detailed_stats['rain_noise']['heavy'] += 1
            else:
                # Light rain - higher frequencies
                rain_adjusted = rain_adjusted.high_pass_filter(1000)
                self.detailed_stats['rain_noise']['light'] += 1
            audio = audio.overlay(rain_adjusted)

        # Fade effect
        elif 0.5 <= random_float < 0.6 and self.noise_stats['fade']<1:
            try:
                if len(audio) > 4000:  # Guard against very short audio
                    fade_duration = int(2000 * random_float2)
                    audio = audio.fade_in(fade_duration).fade_out(fade_duration)
                    self.noise_stats['fade'] += 1
            except Exception as e:
                print(f"Failed to apply fade effect, skipping: {e}")
                # Continue to the next effect

        # Add baby crying sound
        elif 0.6 <= random_float < 0.7 and self.noise_stats['baby_cry']<1:
            self.noise_stats['baby_cry'] += 1
            baby_cry = self.baby_cry[:len(audio)]
            # Adjust cry volume to simulate distance
            cry_volume = random_float2 * 0.8  # Cry volume varies 0-80%
            baby_cry_adjusted = baby_cry - (10 * (1 - cry_volume))
            # Randomly decide continuous or intermittent cry
            if random_float2 > 0.6:
                # Intermittent cry - add only at a random position
                start_pos = int(len(audio) * random_float2 * 0.5)
                end_pos = min(start_pos + int(len(baby_cry_adjusted) * 0.7), len(audio))
                segment = audio[start_pos:end_pos]
                segment = segment.overlay(baby_cry_adjusted[:end_pos-start_pos])
                audio = audio[:start_pos] + segment + audio[end_pos:]
                self.detailed_stats['baby_cry']['intermittent'] += 1
            else:
                # Continuous cry - overlay throughout
                audio = audio.overlay(baby_cry_adjusted)
                self.detailed_stats['baby_cry']['continuous'] += 1

        # Add car horn sound
        elif 0.7 <= random_float < 0.8 and self.noise_stats['car_horn']<1:
            self.noise_stats['car_horn'] += 1
            car = self.car[:len(audio)]
            # Adjust horn volume to simulate distance
            horn_volume = random_float2 * 0.7  # Horn volume varies 0-70%
            car_adjusted = car - (12 * (1 - horn_volume))
            # Randomly decide single or multiple horn blasts
            if random_float2 > 0.5:
                # Multiple horn blasts - add 2-3 times at different positions
                num_horns = 2 + int(random_float2 * 2)
                for i in range(num_horns):
                    start_pos = int(len(audio) * (i + random_float2) / (num_horns + 1))
                    end_pos = min(start_pos + int(len(car) * 0.3), len(audio))
                    segment = audio[start_pos:end_pos]
                    segment = segment.overlay(car_adjusted[:end_pos-start_pos])
                    audio = audio[:start_pos] + segment + audio[end_pos:]
                self.detailed_stats['car_horn']['multiple'] += 1
            else:
                # Single horn blast - add at a random position
                start_pos = int(len(audio) * random_float2)
                audio = audio.overlay(car_adjusted, position=start_pos)
                self.detailed_stats['car_horn']['single'] += 1

        # Add background music
        elif 0.8 <= random_float < 0.9 and self.noise_stats['music']<1:
            self.noise_stats['music'] += 1
            music = self.music[:len(audio)]
            # Adjust music volume as background
            music_volume = random_float2 * 0.6  # Music volume varies 0-60%
            music_adjusted = music - (15 * (1 - music_volume))
            # Randomly decide music type characteristics
            if random_float2 > 0.7:
                # Low-frequency music - simulates bass-heavy music
                music_adjusted = music_adjusted.low_pass_filter(1200)
                self.detailed_stats['music']['low_freq'] += 1
            elif random_float2 > 0.4:
                # Mid-frequency music - keep as is
                self.detailed_stats['music']['mid_freq'] += 1
            else:
                # High-frequency music - simulates treble-heavy music
                music_adjusted = music_adjusted.high_pass_filter(800)
                self.detailed_stats['music']['high_freq'] += 1
            audio = audio.overlay(music_adjusted)

        # Volume adjustment
        elif 0.9 <= random_float <= 1:
            self.noise_stats['volume'] += 1
            if random_float2 > 0 and random_float2 < 0.5 and self.detailed_stats['volume']['up']<6:
                audio = audio + 1 * random_float
                self.detailed_stats['volume']['up'] += 1
            elif self.detailed_stats['volume']['down']<3:
                audio = audio - 0.8 * random_float
                self.detailed_stats['volume']['down'] += 1

        # Prepare output
        try:
            if 0 <= random_float < 0.3:
                # Process this way to prevent distortion
                MAX_INT16 = 32767
                other_audio_int16 = (other_audio * MAX_INT16).astype(np.int16)
                to_audio = AudioSegment(
                    data=other_audio_int16.tobytes(),
                    sample_width=other_audio_int16.dtype.itemsize,
                    frame_rate=sr,
                    channels=1
                )
                to_audio.export(input_path, format="mp3")
            else:
                audio.export(input_path, format="mp3")
        except Exception as e:
            print(f"Error: Failed to export audio file - {e}")
            return None, None

        return random_float, random_float2  # Return the noise parameters used for logging

# Provide a global instance and function for backwards compatibility
noise_adder = NoiseAdder()

def add_noise(input_path, way=None, parameter=None):
    """Compatibility function for legacy code."""
    return noise_adder.add_noise(input_path, way, parameter)
