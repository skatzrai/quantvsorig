import os
import zipfile
import time
import traceback
from pathlib import Path
from functools import lru_cache
from typing import Optional

import gdown
import huggingface_hub
import numpy as np
import psutil
import pandas as pd
import streamlit as st
import onnxruntime as ort
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

st.set_page_config(page_title="Quantized model tester", layout="wide")

# ============================================================
# 🔥 QuantModel (встроен сюда, с корректным пуллингом токенов)
# ============================================================
class QuantModel:
    """Универсальный загрузчик квантизированных ONNX моделей."""
    def __init__(self, model_id: str, source: str = "gdrive",
                 model_dir: str = "onnx_model", tokenizer_name: Optional[str] = None,
                 force_download: bool = False):
        self.model_id = model_id
        self.source = source
        self.model_dir = Path(model_dir)
        self.tokenizer_name = tokenizer_name
        self.force_download = force_download
        self.model_path = None

        self._ensure_model()
        self.session = self._load_session()
        self.tokenizer = self._load_tokenizer()

    def _ensure_model(self):
        os.makedirs(self.model_dir, exist_ok=True)
        need_download = self.force_download or not any(self.model_dir.glob("*.onnx"))

        if need_download:
            if self.source == "gdrive":
                zip_path = f"{self.model_dir}.zip"
                gdown.download(f"https://drive.google.com/uc?id={self.model_id}", zip_path, quiet=False)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(self.model_dir)
                os.remove(zip_path)
            elif self.source == "hf":
                huggingface_hub.snapshot_download(
                    repo_id=self.model_id,
                    local_dir=self.model_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True
                )
            elif self.source == "local":
                # ничего не качаем — берем как есть
                pass
            else:
                raise ValueError(f"❌ Неизвестный источник: {self.source}")

        onnx_files = list(self.model_dir.rglob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"❌ Нет .onnx модели в {self.model_dir}")
        self.model_path = onnx_files[0]

    def _load_session(self):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        try:
            if ort.get_device() == "GPU":
                providers.insert(0, "CUDAExecutionProvider")
        except Exception:
            pass
        return ort.InferenceSession(str(self.model_path), sess_options=so, providers=providers)

    def _load_tokenizer(self):
        if self.tokenizer_name:
            return AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
        try:
            return AutoTokenizer.from_pretrained(str(self.model_dir), use_fast=True)
        except Exception:
            return AutoTokenizer.from_pretrained("deepvk/USER-BGE-M3", use_fast=True)

    @lru_cache(maxsize=1024)
    def _encode_cached(self, text: str, normalize: bool = True):
        # Токенизация
        inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors="np")
        ort_inputs = {k: v for k, v in inputs.items()}
        # Прогон
        outputs = self.session.run(None, ort_inputs)
        embeddings = outputs[0]

        # Унификация формы:
        # - если (batch, seq, dim) -> делаем masked mean по seq (attention_mask)
        # - если (batch, dim) -> уже sentence embedding
        if embeddings.ndim == 3:
            # masked mean pooling
            mask = ort_inputs.get("attention_mask", None)
            if mask is not None:
                mask = mask.astype(np.float32)[..., None]  # (batch, seq, 1)
                summed = (embeddings * mask).sum(axis=1)   # (batch, dim)
                counts = mask.sum(axis=1)                  # (batch, 1)
                counts = np.clip(counts, 1e-6, None)
                embeddings = summed / counts
            else:
                embeddings = embeddings.mean(axis=1)

        # Нормализация на уровне предложений
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
            embeddings = embeddings / norms

        # Возвращаем (dim,) для одного текста
        return embeddings[0]

    def encode(self, texts, normalize=True):
        if isinstance(texts, str):
            texts = [texts]
        return np.array([self._encode_cached(t, normalize) for t in texts])


# ============================================================
# 🔧 Helpers
# ============================================================
def to_vector(embs):
    """Превращает батч эмбеддингов в один вектор (усреднение)."""
    arr = np.array(embs)
    # (dim,)
    if arr.ndim == 1:
        return arr
    # (1, dim)
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr[0]
    # (batch, seq, dim) -> среднее по seq, затем по batch
    if arr.ndim == 3:
        arr = arr.mean(axis=1)
    # (batch, dim) -> среднее по batch
    return arr.mean(axis=0)


def cosine_similarity(vec1, vec2):
    return float(np.dot(vec1, vec2) / ((norm(vec1) * norm(vec2)) + 1e-12))


def cosine_batch(A, B):
    """Поэлементный косинус для пар предложений."""
    A = np.asarray(A)
    B = np.asarray(B)
    if A.shape != B.shape:
        # подрезаем до min(dim)
        m = min(A.shape[-1], B.shape[-1])
        A = A[..., :m]
        B = B[..., :m]
    # считаем, что уже нормализованы; но на всякий случай повторим
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return (A * B).sum(axis=1)


# ============================================================
# 🎛️ UI
# ============================================================
st.title("🔍 Тестирование моделей: Оригинал vs Квант")

mode = st.radio("Выберите режим:", ["Оригинальная модель", "Квантованная модель", "Сравнение обеих"])

# Кнопка сброса сессии/кэшей
if st.button("♻️ Сбросить сессию"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

input_text = st.text_area("Тексты для теста (по одной строке)", "Это тестовое предложение.\nПример второй строки.")
texts = [t.strip() for t in input_text.split("\n") if t.strip()]
batch_size = st.slider("Количество повторов для throughput-теста", 1, 128, 8)
force_download = st.checkbox("♻️ Перекачать квант-модель заново", False)

metrics_df = None

if mode == "Оригинальная модель":
    model_id = st.text_input("HF repo ID", "deepvk/USER-BGE-M3")

elif mode == "Квантованная модель":
    col1, col2 = st.columns(2)
    with col1:
        quant_source = st.selectbox("Источник", ["gdrive", "hf", "local"], index=1)
        quant_id = st.text_input("ID/Repo/Path", "1lkrvCPIE1wvffIuCSHGtbEz3Epjx5R36")
    with col2:
        quant_dir = st.text_input("Папка для кванта", "onnx-user-bge-m3")
        tokenizer_name = st.text_input("Tokenizer name", "")

else:  # Сравнение обеих
    st.markdown("В этом режиме измеряем **только качество** (cosine similarity) и скорость. Память не меряем.")
    col1, col2 = st.columns(2)
    with col1:
        model_id = st.text_input("HF repo ID (оригинал)", "deepvk/USER-BGE-M3", key="orig_repo_cmp")
    with col2:
        quant_source = st.selectbox("Источник кванта", ["gdrive", "hf", "local"], index=1, key="quant_src_cmp")
        quant_id = st.text_input("ID/Repo/Path (квант)", "1lkrvCPIE1wvffIuCSHGtbEz3Epjx5R36", key="quant_id_cmp")
    col3, col4 = st.columns(2)
    with col3:
        quant_dir = st.text_input("Папка для кванта", "onnx-user-bge-m3", key="quant_dir_cmp")
    with col4:
        tokenizer_name = st.text_input("Tokenizer name", "", key="tok_cmp")

run_button = st.button("🚀 Запустить тест")

# ============================================================
# 🚀 Запуск теста
# ============================================================
if run_button:
    try:
        texts_for_run = (texts * batch_size)[:max(len(texts), 1)]

        if mode == "Оригинальная модель":
            proc = psutil.Process()
            with st.spinner("Загрузка оригинальной модели..."):
                model = SentenceTransformer(model_id)
            t0 = time.perf_counter()
            embs = model.encode(texts_for_run, normalize_embeddings=True)
            t1 = time.perf_counter()
            latency = t1 - t0
            memory = proc.memory_info().rss / 1024 ** 2

            metrics_df = pd.DataFrame([{
                "Mode": "Original",
                "Batch Size": batch_size,
                "Latency (s)": latency,
                "Throughput (texts/s)": len(texts_for_run) / max(latency, 1e-12),
                "Memory (MB)": memory
            }])

            st.subheader("📊 Результаты")
            st.metric("Latency (s)", f"{latency:.4f}")
            st.metric("Throughput (texts/s)", f"{len(texts_for_run)/max(latency,1e-12):.1f}")
            st.metric("Memory (MB)", f"{memory:.1f}")
            st.dataframe(metrics_df)

        elif mode == "Квантованная модель":
            proc = psutil.Process()
            with st.spinner("Загрузка квантованной модели..."):
                model = QuantModel(
                    model_id=quant_id,
                    source=quant_source,
                    model_dir=quant_dir,
                    tokenizer_name=tokenizer_name if tokenizer_name else None,
                    force_download=force_download
                )
            t0 = time.perf_counter()
            embs = model.encode(texts_for_run, normalize=True)
            t1 = time.perf_counter()
            latency = t1 - t0
            memory = proc.memory_info().rss / 1024 ** 2

            metrics_df = pd.DataFrame([{
                "Mode": "Quantized",
                "Batch Size": batch_size,
                "Latency (s)": latency,
                "Throughput (texts/s)": len(texts_for_run) / max(latency, 1e-12),
                "Memory (MB)": memory
            }])

            st.subheader("📊 Результаты")
            st.metric("Latency (s)", f"{latency:.4f}")
            st.metric("Throughput (texts/s)", f"{len(texts_for_run)/max(latency,1e-12):.1f}")
            st.metric("Memory (MB)", f"{memory:.1f}")
            st.dataframe(metrics_df)

        else:  # Сравнение обеих
            # Original
            with st.spinner("Загрузка оригинальной модели..."):
                orig = SentenceTransformer(model_id)
            t0 = time.perf_counter()
            orig_embs = orig.encode(texts_for_run, normalize_embeddings=True)
            t1 = time.perf_counter()
            orig_latency = t1 - t0

            # Quantized
            with st.spinner("Загрузка квантованной модели..."):
                quant = QuantModel(
                    model_id=quant_id,
                    source=quant_source,
                    model_dir=quant_dir,
                    tokenizer_name=tokenizer_name if tokenizer_name else None,
                    force_download=force_download
                )
            t0 = time.perf_counter()
            quant_embs = quant.encode(texts_for_run, normalize=True)
            t1 = time.perf_counter()
            quant_latency = t1 - t0

            # Приведение форм (на всякий случай)
            O = np.asarray(orig_embs)
            Q = np.asarray(quant_embs)
            if O.ndim == 3:
                O = O.mean(axis=1)
            if Q.ndim == 3:
                Q = Q.mean(axis=1)
            if O.shape[1] != Q.shape[1]:
                m = min(O.shape[1], Q.shape[1])
                O = O[:, :m]
                Q = Q[:, :m]

            # Косинусы по каждому тексту + среднее
            per_text_cos = cosine_batch(O, Q)
            avg_cos = float(per_text_cos.mean())

            # Таблица метрик (без памяти)
            metrics_df = pd.DataFrame([
                {
                    "Mode": "Original",
                    "Batch Size": batch_size,
                    "Latency (s)": orig_latency,
                    "Throughput (texts/s)": len(texts_for_run) / max(orig_latency, 1e-12),
                },
                {
                    "Mode": "Quantized",
                    "Batch Size": batch_size,
                    "Latency (s)": quant_latency,
                    "Throughput (texts/s)": len(texts_for_run) / max(quant_latency, 1e-12),
                },
            ])

            st.subheader("📊 Метрики скорости")
            st.dataframe(metrics_df)
            st.subheader("🎯 Качество")
            st.write(f"Средняя cosine similarity (по {len(per_text_cos)} текстам): **{avg_cos:.4f}**")

        # Кнопка выгрузки CSV (для любого режима)
        if metrics_df is not None:
            csv = metrics_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Скачать результаты (CSV)",
                data=csv,
                file_name="metrics.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Ошибка: {e}")
        st.text(traceback.format_exc())
