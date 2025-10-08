import os
import soundfile as sf

input_dir = "data/flac_data"
output_dir = "data/samples"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".flac"):
        flac_path = os.path.join(input_dir, filename)
        wav_path = os.path.join(output_dir, filename.replace(".flac", ".wav"))

        data, samplerate = sf.read(flac_path)
        sf.write(wav_path, data, samplerate)

        print(f"已转换: {filename}")
