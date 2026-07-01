import joblib
encoder = joblib.load("encoder.pkl")
print("Encoder classes:", encoder.classes_)
