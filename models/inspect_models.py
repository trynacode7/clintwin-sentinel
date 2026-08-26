import torch
import joblib

print("=" * 50)
print("LSTM MODEL")
print("=" * 50)

lstm = torch.load("lstm_model.pt", map_location="cpu", weights_only=False)
print("Type:", type(lstm))

if isinstance(lstm, dict):
    print("Keys:", list(lstm.keys()))
else:
    print(lstm)


print("\n" + "=" * 50)
print("TRANSFORMER MODEL")
print("=" * 50)

transformer = torch.load(
    "transformer_model.pt",
    map_location="cpu",
    weights_only=False
)

print("Type:", type(transformer))

if isinstance(transformer, dict):
    print("Keys:", list(transformer.keys()))
else:
    print(transformer)


print("\n" + "=" * 50)
print("SCALER")
print("=" * 50)

scaler = joblib.load("scaler.joblib")

print("Type:", type(scaler))
print(scaler)