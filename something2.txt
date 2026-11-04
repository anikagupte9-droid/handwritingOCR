from transformers import AutoTokenizer
import json

tok = AutoTokenizer.from_pretrained("onnx_int8")

vocab = tok.get_vocab()

id_to_token = {
    str(token_id): token
    for token, token_id in vocab.items()
}

with open("vocab_android.json", "w", encoding="utf-8") as f:
    json.dump(
        id_to_token,
        f,
        ensure_ascii=False,
        indent=2
    )

print("Created vocab_android.json")
print("Entries:", len(id_to_token))