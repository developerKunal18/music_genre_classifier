import librosa

def extract_features(file):
    audio, sr = librosa.load(file)

    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr).mean()
    zero_cross_rate = librosa.feature.zero_crossing_rate(audio).mean()

    return tempo, spectral_centroid, zero_cross_rate

def classify_genre(tempo, centroid, zcr):
    if tempo > 120 and zcr > 0.1:
        return "🎶 Pop"
    elif centroid > 3000 and zcr > 0.15:
        return "🎸 Rock"
    elif tempo < 90 and centroid < 2000:
        return "🎻 Classical"
    elif tempo > 140 and zcr > 0.18:
        return "🎤 Hip-Hop"
    else:
        return "🎷 Jazz"

def main():
    print("🎧 Music Genre Classifier")
    file = input("Enter audio file path: ")

    try:
        tempo, centroid, zcr = extract_features(file)
        genre = classify_genre(tempo, centroid, zcr)

        print("\n✅ Analysis Complete!")
        print(f"Estimated Genre: {genre}")
        print(f"Tempo (BPM): {tempo:.2f}")
        print(f"Spectral Centroid: {centroid:.2f}")
        print(f"Zero-Crossing Rate: {zcr:.4f}")

    except Exception as e:
        print("❌ Error:", e)
        print("Make sure the file exists and is a valid audio format.")

if __name__ == "__main__":
    main()
