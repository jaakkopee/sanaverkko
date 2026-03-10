import argparse
import os
import pickle
import re
import sys
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None


def _resolve_torch_device(preferred="auto"):
    if torch is None:
        return None

    torch_module = torch

    preferred_value = str(preferred or "auto").lower().strip()
    if preferred_value == "auto":
        if torch_module.cuda.is_available():
            return "cuda"
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return "mps"
        return "cpu"

    if preferred_value == "cuda":
        return "cuda" if torch_module.cuda.is_available() else None
    if preferred_value == "mps":
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return "mps"
        return None
    if preferred_value == "cpu":
        return "cpu"
    return None


def _tokenize_text(text):
    return re.findall(r"[a-zåäö]+", text.lower())


class CharCNNEncoder:
    def __init__(self, alphabet, embedding_dim=32, filters_per_width=64, widths=(2, 3, 4), max_word_len=24, seed=1234):
        self.alphabet = list(alphabet)
        self.alphabet_index = {char: index + 1 for index, char in enumerate(self.alphabet)}
        self.embedding_dim = int(embedding_dim)
        self.filters_per_width = int(filters_per_width)
        self.widths = tuple(int(width) for width in widths)
        self.max_word_len = int(max_word_len)

        rng = np.random.default_rng(seed)
        vocab_chars = len(self.alphabet) + 1
        self.embedding = rng.normal(0.0, 0.08, size=(vocab_chars, self.embedding_dim)).astype(np.float32)
        self.conv_weights = {}
        self.conv_bias = {}
        for width in self.widths:
            self.conv_weights[width] = rng.normal(
                0.0,
                0.08,
                size=(self.filters_per_width, width, self.embedding_dim),
            ).astype(np.float32)
            self.conv_bias[width] = np.zeros((self.filters_per_width,), dtype=np.float32)

    @property
    def output_dim(self):
        return len(self.widths) * self.filters_per_width

    def _word_to_ids(self, word):
        ids = np.zeros((self.max_word_len,), dtype=np.int32)
        truncated = word[: self.max_word_len]
        for i, char in enumerate(truncated):
            ids[i] = self.alphabet_index.get(char, 0)
        return ids

    def encode_word(self, word):
        ids = self._word_to_ids(word)
        embedded = self.embedding[ids]
        pooled = []
        for width in self.widths:
            kernel = self.conv_weights[width]
            bias = self.conv_bias[width]
            steps = self.max_word_len - width + 1
            if steps <= 0:
                pooled.append(np.zeros((self.filters_per_width,), dtype=np.float32))
                continue

            conv_out = np.empty((steps, self.filters_per_width), dtype=np.float32)
            for position in range(steps):
                window = embedded[position : position + width]
                conv_out[position] = np.tensordot(kernel, window, axes=((1, 2), (0, 1))) + bias

            conv_out = np.maximum(conv_out, 0.0)
            pooled.append(conv_out.max(axis=0))

        return np.concatenate(pooled, axis=0)


class SVLTMModel:
    def __init__(self, model_data):
        self.version = model_data["version"]
        self.topology = model_data["topology"]
        self.context_size = int(model_data["context_size"])
        self.device_preference = str(model_data.get("device_preference", "auto"))

        self.alphabet = model_data["alphabet"]
        self.word_to_index = model_data["word_to_index"]
        self.index_to_word = model_data["index_to_word"]

        self.encoder = CharCNNEncoder(
            alphabet=self.alphabet,
            embedding_dim=int(model_data["encoder"]["embedding_dim"]),
            filters_per_width=int(model_data["encoder"]["filters_per_width"]),
            widths=tuple(model_data["encoder"]["widths"]),
            max_word_len=int(model_data["encoder"]["max_word_len"]),
            seed=int(model_data["encoder"]["seed"]),
        )
        self.encoder.embedding = model_data["encoder"]["embedding"].astype(np.float32)
        for width in self.encoder.widths:
            self.encoder.conv_weights[width] = model_data["encoder"]["conv_weights"][width].astype(np.float32)
            self.encoder.conv_bias[width] = model_data["encoder"]["conv_bias"][width].astype(np.float32)

        self.W1 = model_data["mlp"]["W1"].astype(np.float32)
        self.b1 = model_data["mlp"]["b1"].astype(np.float32)
        self.W2 = model_data["mlp"]["W2"].astype(np.float32)
        self.b2 = model_data["mlp"]["b2"].astype(np.float32)
        self.W3 = model_data["mlp"]["W3"].astype(np.float32)
        self.b3 = model_data["mlp"]["b3"].astype(np.float32)

        self.word_feature_cache = model_data.get("word_feature_cache", {})

        self._torch_inference_device = None
        self._torch_inference_enabled = False
        self._torch_W1 = None
        self._torch_b1 = None
        self._torch_W2 = None
        self._torch_b2 = None
        self._torch_W3 = None
        self._torch_b3 = None
        self._setup_torch_inference(self.device_preference)

    def _setup_torch_inference(self, preferred_device):
        resolved_device = _resolve_torch_device(preferred_device)
        if torch is None or resolved_device is None or resolved_device == "cpu":
            self._torch_inference_enabled = False
            self._torch_inference_device = None
            return

        assert torch is not None

        try:
            device = torch.device(resolved_device)
            self._torch_W1 = torch.from_numpy(self.W1).to(device)
            self._torch_b1 = torch.from_numpy(self.b1).to(device)
            self._torch_W2 = torch.from_numpy(self.W2).to(device)
            self._torch_b2 = torch.from_numpy(self.b2).to(device)
            self._torch_W3 = torch.from_numpy(self.W3).to(device)
            self._torch_b3 = torch.from_numpy(self.b3).to(device)
            self._torch_inference_device = device
            self._torch_inference_enabled = True
        except Exception:
            self._torch_inference_enabled = False
            self._torch_inference_device = None

    def runtime_backend(self):
        if self._torch_inference_enabled and self._torch_inference_device is not None:
            try:
                return str(self._torch_inference_device.type)
            except Exception:
                return "gpu"
        return "cpu"

    @staticmethod
    def _relu(x):
        return np.maximum(x, 0.0)

    @staticmethod
    def _softmax(logits):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    @staticmethod
    def _build_alphabet(words):
        alphabet = sorted({char for word in words for char in word})
        return alphabet

    @staticmethod
    def _build_vocab(words, min_count=1):
        counts = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1

        filtered = [word for word, count in counts.items() if count >= min_count]
        filtered.sort()

        word_to_index = {word: index for index, word in enumerate(filtered)}
        index_to_word = filtered
        return word_to_index, index_to_word

    @classmethod
    def train_from_words(
        cls,
        words,
        context_size=3,
        epochs=10,
        learning_rate=0.02,
        l2=1e-5,
        hidden1=1024,
        hidden2=512,
        min_word_count=1,
        embedding_dim=32,
        filters_per_width=64,
        widths=(2, 3, 4, 5),
        max_word_len=24,
        seed=1234,
        batch_size=64,
        device="auto",
        verbose=True,
    ):
        if len(words) <= context_size:
            raise ValueError("Not enough words for training data")

        context_size = int(context_size)
        alphabet = cls._build_alphabet(words)
        if not alphabet:
            raise ValueError("No valid alphabet characters found")

        word_to_index, index_to_word = cls._build_vocab(words, min_count=min_word_count)
        if len(word_to_index) < 2:
            raise ValueError("Vocabulary too small after filtering")

        encoder = CharCNNEncoder(
            alphabet=alphabet,
            embedding_dim=embedding_dim,
            filters_per_width=filters_per_width,
            widths=widths,
            max_word_len=max_word_len,
            seed=seed,
        )

        vocab_size = len(index_to_word)
        feature_dim = encoder.output_dim * context_size

        rng = np.random.default_rng(seed)
        W1 = rng.normal(0.0, 0.05, size=(feature_dim, int(hidden1))).astype(np.float32)
        b1 = np.zeros((int(hidden1),), dtype=np.float32)
        W2 = rng.normal(0.0, 0.05, size=(int(hidden1), int(hidden2))).astype(np.float32)
        b2 = np.zeros((int(hidden2),), dtype=np.float32)
        W3 = rng.normal(0.0, 0.05, size=(int(hidden2), vocab_size)).astype(np.float32)
        b3 = np.zeros((vocab_size,), dtype=np.float32)

        word_feature_cache = {word: encoder.encode_word(word) for word in index_to_word}
        zero_feature = np.zeros((encoder.output_dim,), dtype=np.float32)

        samples = []
        targets = []
        for index in range(context_size, len(words)):
            target_word = words[index]
            if target_word not in word_to_index:
                continue
            context_words = words[index - context_size : index]
            samples.append(context_words)
            targets.append(word_to_index[target_word])

        if not samples:
            raise ValueError("No trainable samples after vocabulary filtering")

        y = np.array(targets, dtype=np.int32)

        def make_context_feature(context_words):
            vectors = []
            for word in context_words:
                vectors.append(word_feature_cache.get(word, zero_feature))
            if len(vectors) < context_size:
                vectors = [zero_feature] * (context_size - len(vectors)) + vectors
            return np.concatenate(vectors, axis=0)

        X = np.stack([make_context_feature(context_words) for context_words in samples], axis=0).astype(np.float32)

        def relu(x):
            return np.maximum(x, 0.0)

        def softmax(logits):
            shifted = logits - np.max(logits, axis=1, keepdims=True)
            exp_values = np.exp(shifted)
            return exp_values / np.sum(exp_values, axis=1, keepdims=True)

        sample_count = X.shape[0]
        batch_size_local = max(8, int(batch_size))
        total_batches = max(1, (sample_count + batch_size_local - 1) // batch_size_local)

        resolved_device = _resolve_torch_device(device)
        use_torch_training = torch is not None and F is not None and resolved_device in {"cuda", "mps"}

        if use_torch_training:
            assert torch is not None
            assert F is not None
            if verbose:
                print(f"Using torch backend on {resolved_device} for training")

            device_obj = torch.device(resolved_device)
            W1_t = torch.tensor(W1, device=device_obj, requires_grad=True)
            b1_t = torch.tensor(b1, device=device_obj, requires_grad=True)
            W2_t = torch.tensor(W2, device=device_obj, requires_grad=True)
            b2_t = torch.tensor(b2, device=device_obj, requires_grad=True)
            W3_t = torch.tensor(W3, device=device_obj, requires_grad=True)
            b3_t = torch.tensor(b3, device=device_obj, requires_grad=True)

            optimizer = torch.optim.SGD([W1_t, b1_t, W2_t, b2_t, W3_t, b3_t], lr=float(learning_rate))

            for epoch in range(max(1, int(epochs))):
                order = np.arange(sample_count)
                rng.shuffle(order)

                epoch_loss = 0.0
                processed_samples = 0
                for batch_index, start in enumerate(range(0, sample_count, batch_size_local), start=1):
                    batch_indices = order[start : start + batch_size_local]
                    xb = torch.from_numpy(X[batch_indices]).to(device_obj)
                    yb = torch.from_numpy(y[batch_indices]).to(device_obj, dtype=torch.long)

                    optimizer.zero_grad(set_to_none=True)

                    z1 = xb @ W1_t + b1_t
                    a1 = torch.relu(z1)
                    z2 = a1 @ W2_t + b2_t
                    a2 = torch.relu(z2)
                    logits = a2 @ W3_t + b3_t

                    ce_loss = F.cross_entropy(logits, yb)
                    reg_loss = 0.5 * float(l2) * (
                        torch.sum(W1_t * W1_t) + torch.sum(W2_t * W2_t) + torch.sum(W3_t * W3_t)
                    )
                    loss = ce_loss + reg_loss

                    loss.backward()
                    optimizer.step()

                    batch_size_now = int(xb.shape[0])
                    epoch_loss += float(loss.detach().cpu().item()) * batch_size_now
                    processed_samples += batch_size_now

                    if verbose:
                        progress = batch_index / float(total_batches)
                        bar_width = 32
                        filled = int(bar_width * progress)
                        bar = "=" * filled + "-" * (bar_width - filled)
                        running_loss = epoch_loss / float(max(1, processed_samples))
                        sys.stdout.write(
                            f"\repoch {epoch + 1}/{epochs} [{bar}] {batch_index}/{total_batches} {progress * 100:6.2f}% loss~{running_loss:.6f}"
                        )
                        sys.stdout.flush()

                if verbose:
                    final_bar = "=" * 32
                    final_loss = epoch_loss / float(max(1, sample_count))
                    sys.stdout.write(
                        f"\repoch {epoch + 1}/{epochs} [{final_bar}] {total_batches}/{total_batches} 100.00% loss={final_loss:.6f}\n"
                    )
                    sys.stdout.flush()

            W1 = W1_t.detach().cpu().numpy().astype(np.float32)
            b1 = b1_t.detach().cpu().numpy().astype(np.float32)
            W2 = W2_t.detach().cpu().numpy().astype(np.float32)
            b2 = b2_t.detach().cpu().numpy().astype(np.float32)
            W3 = W3_t.detach().cpu().numpy().astype(np.float32)
            b3 = b3_t.detach().cpu().numpy().astype(np.float32)
        else:
            for epoch in range(max(1, int(epochs))):
                order = np.arange(sample_count)
                rng.shuffle(order)

                epoch_loss = 0.0
                processed_samples = 0
                for batch_index, start in enumerate(range(0, sample_count, batch_size_local), start=1):
                    batch_indices = order[start : start + batch_size_local]
                    xb = X[batch_indices]
                    yb = y[batch_indices]

                    z1 = xb @ W1 + b1
                    a1 = relu(z1)
                    z2 = a1 @ W2 + b2
                    a2 = relu(z2)
                    logits = a2 @ W3 + b3
                    probs = softmax(logits)

                    batch_size_now = xb.shape[0]
                    loss = -np.log(probs[np.arange(batch_size_now), yb] + 1e-9).mean()
                    loss += 0.5 * l2 * (
                        np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3)
                    )
                    epoch_loss += loss * batch_size_now
                    processed_samples += batch_size_now

                    d_logits = probs
                    d_logits[np.arange(batch_size_now), yb] -= 1.0
                    d_logits /= float(batch_size_now)

                    dW3 = a2.T @ d_logits + l2 * W3
                    db3 = np.sum(d_logits, axis=0)

                    d_a2 = d_logits @ W3.T
                    d_z2 = d_a2 * (z2 > 0.0)
                    dW2 = a1.T @ d_z2 + l2 * W2
                    db2 = np.sum(d_z2, axis=0)

                    d_a1 = d_z2 @ W2.T
                    d_z1 = d_a1 * (z1 > 0.0)
                    dW1 = xb.T @ d_z1 + l2 * W1
                    db1 = np.sum(d_z1, axis=0)

                    W3 -= learning_rate * dW3
                    b3 -= learning_rate * db3
                    W2 -= learning_rate * dW2
                    b2 -= learning_rate * db2
                    W1 -= learning_rate * dW1
                    b1 -= learning_rate * db1

                    if verbose:
                        progress = batch_index / float(total_batches)
                        bar_width = 32
                        filled = int(bar_width * progress)
                        bar = "=" * filled + "-" * (bar_width - filled)
                        running_loss = epoch_loss / float(max(1, processed_samples))
                        sys.stdout.write(
                            f"\repoch {epoch + 1}/{epochs} [{bar}] {batch_index}/{total_batches} {progress * 100:6.2f}% loss~{running_loss:.6f}"
                        )
                        sys.stdout.flush()

                if verbose:
                    final_bar = "=" * 32
                    final_loss = epoch_loss / float(max(1, sample_count))
                    sys.stdout.write(
                        f"\repoch {epoch + 1}/{epochs} [{final_bar}] {total_batches}/{total_batches} 100.00% loss={final_loss:.6f}\n"
                    )
                    sys.stdout.flush()

        model_data = {
            "version": 1,
            "topology": "char_cnn_mlp_word_softmax_ctx3",
            "context_size": context_size,
            "device_preference": str(device),
            "alphabet": alphabet,
            "word_to_index": word_to_index,
            "index_to_word": index_to_word,
            "encoder": {
                "embedding_dim": int(embedding_dim),
                "filters_per_width": int(filters_per_width),
                "widths": tuple(int(width) for width in widths),
                "max_word_len": int(max_word_len),
                "seed": int(seed),
                "embedding": encoder.embedding,
                "conv_weights": encoder.conv_weights,
                "conv_bias": encoder.conv_bias,
            },
            "mlp": {
                "W1": W1,
                "b1": b1,
                "W2": W2,
                "b2": b2,
                "W3": W3,
                "b3": b3,
            },
            "word_feature_cache": word_feature_cache,
        }
        return cls(model_data)

    def _context_vector(self, context_words):
        zero = np.zeros((self.encoder.output_dim,), dtype=np.float32)
        context = list(context_words)[-self.context_size :]
        if len(context) < self.context_size:
            context = [""] * (self.context_size - len(context)) + context

        vectors = []
        for word in context:
            if not word:
                vectors.append(zero)
                continue
            cached = self.word_feature_cache.get(word)
            if cached is None:
                cached = self.encoder.encode_word(word)
            vectors.append(cached)
        return np.concatenate(vectors, axis=0).astype(np.float32)

    def _forward_logits(self, context_words):
        if self._torch_inference_enabled and self._torch_inference_device is not None:
            assert torch is not None
            x_np = self._context_vector(context_words).reshape(1, -1)
            with torch.no_grad():
                x = torch.from_numpy(x_np).to(self._torch_inference_device)
                a1 = torch.relu(x @ self._torch_W1 + self._torch_b1)
                a2 = torch.relu(a1 @ self._torch_W2 + self._torch_b2)
                logits = a2 @ self._torch_W3 + self._torch_b3
            return logits.detach().cpu().numpy()

        x = self._context_vector(context_words).reshape(1, -1)
        a1 = self._relu(x @ self.W1 + self.b1)
        a2 = self._relu(a1 @ self.W2 + self.b2)
        return a2 @ self.W3 + self.b3

    def predict_next_probabilities(self, context_words, candidate_words=None):
        logits = self._forward_logits(context_words)
        probs = self._softmax(logits)[0]

        if candidate_words is None:
            result = {self.index_to_word[index]: float(probs[index]) for index in range(len(self.index_to_word))}
            return result

        selected = {}
        total = 0.0
        for word in candidate_words:
            index = self.word_to_index.get(word)
            probability = float(probs[index]) if index is not None else 0.0
            selected[word] = probability
            total += probability

        if total > 0.0:
            for word in selected:
                selected[word] /= total
        return selected

    def save(self, file_path):
        if not str(file_path).lower().endswith(".svltm"):
            raise ValueError("Model file must use .svltm suffix")
        payload = {
            "version": self.version,
            "topology": self.topology,
            "context_size": self.context_size,
            "device_preference": self.device_preference,
            "alphabet": self.alphabet,
            "word_to_index": self.word_to_index,
            "index_to_word": self.index_to_word,
            "encoder": {
                "embedding_dim": self.encoder.embedding_dim,
                "filters_per_width": self.encoder.filters_per_width,
                "widths": self.encoder.widths,
                "max_word_len": self.encoder.max_word_len,
                "seed": 0,
                "embedding": self.encoder.embedding,
                "conv_weights": self.encoder.conv_weights,
                "conv_bias": self.encoder.conv_bias,
            },
            "mlp": {
                "W1": self.W1,
                "b1": self.b1,
                "W2": self.W2,
                "b2": self.b2,
                "W3": self.W3,
                "b3": self.b3,
            },
            "word_feature_cache": self.word_feature_cache,
        }
        with open(file_path, "wb") as output_file:
            pickle.dump(payload, output_file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, file_path):
        with open(file_path, "rb") as input_file:
            payload = pickle.load(input_file)
        return cls(payload)


def train_from_text_file(
    input_text_file,
    output_model_file,
    context_size=3,
    epochs=10,
    learning_rate=0.02,
    hidden1=1024,
    hidden2=512,
    min_word_count=1,
    embedding_dim=32,
    filters_per_width=64,
    widths=(2, 3, 4, 5),
    max_word_len=24,
    seed=1234,
    batch_size=64,
    device="auto",
):
    with open(input_text_file, "r", encoding="utf-8") as input_file:
        text = input_file.read()

    words = _tokenize_text(text)
    if len(words) <= int(context_size):
        raise ValueError("Not enough words for requested context size")

    model = SVLTMModel.train_from_words(
        words=words,
        context_size=context_size,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden1=hidden1,
        hidden2=hidden2,
        min_word_count=min_word_count,
        embedding_dim=embedding_dim,
        filters_per_width=filters_per_width,
        widths=widths,
        max_word_len=max_word_len,
        seed=seed,
        batch_size=batch_size,
        device=device,
        verbose=True,
    )
    model.save(output_model_file)
    return model


def _normalize_input_files(input_text_files):
    if isinstance(input_text_files, str):
        raw_values = [input_text_files]
    else:
        raw_values = list(input_text_files or [])

    normalized = []
    for value in raw_values:
        if not isinstance(value, str):
            continue
        parts = [item.strip() for item in value.split(",")]
        for part in parts:
            if part != "":
                normalized.append(part)

    return normalized


def train_from_text_files(
    input_text_files,
    output_model_file,
    context_size=3,
    epochs=10,
    learning_rate=0.02,
    hidden1=1024,
    hidden2=512,
    min_word_count=1,
    embedding_dim=32,
    filters_per_width=64,
    widths=(2, 3, 4, 5),
    max_word_len=24,
    seed=1234,
    batch_size=64,
    device="auto",
):
    input_files = _normalize_input_files(input_text_files)
    if not input_files:
        raise ValueError("At least one input text file is required")

    missing_files = [path for path in input_files if not os.path.isfile(path)]
    if missing_files:
        raise FileNotFoundError(f"Input file(s) not found: {', '.join(missing_files)}")

    words = []
    for input_file_path in input_files:
        with open(input_file_path, "r", encoding="utf-8") as input_file:
            text = input_file.read()
        words.extend(_tokenize_text(text))

    if len(words) <= int(context_size):
        raise ValueError("Not enough words for requested context size")

    model = SVLTMModel.train_from_words(
        words=words,
        context_size=context_size,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden1=hidden1,
        hidden2=hidden2,
        min_word_count=min_word_count,
        embedding_dim=embedding_dim,
        filters_per_width=filters_per_width,
        widths=widths,
        max_word_len=max_word_len,
        seed=seed,
        batch_size=batch_size,
        device=device,
        verbose=True,
    )
    model.save(output_model_file)
    return model


def load_model(model_file):
    return SVLTMModel.load(model_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, action="append", help="Input text file(s). Repeat --input for multiple files or use comma-separated values.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--context", type=int, default=3)
    parser.add_argument("--hidden1", type=int, default=1024)
    parser.add_argument("--hidden2", type=int, default=512)
    parser.add_argument("--min-word-count", type=int, default=1)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--filters", type=int, default=64)
    parser.add_argument("--max-word-len", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    output_file = args.output
    if not output_file.lower().endswith(".svltm"):
        output_file = f"{output_file}.svltm"

    train_from_text_files(
        input_text_files=args.input,
        output_model_file=output_file,
        context_size=args.context,
        epochs=args.epochs,
        learning_rate=args.lr,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        min_word_count=args.min_word_count,
        embedding_dim=args.embedding_dim,
        filters_per_width=args.filters,
        widths=(2, 3, 4, 5),
        max_word_len=args.max_word_len,
        seed=args.seed,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
