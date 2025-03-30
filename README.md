# rag-local-intel-cpu

Setup a venv.

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies for langchain app.

```bash
pip install --upgrade --quiet \
  langchain \
  langchain_community \
  langchain_huggingface \
  lxml[html_clean] \
  bs4 \
  psycopg \
  pgvector
```

Install dependencies for llm serving with llama-cpp-python.

```bash
pip install --quiet \
  llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

Install extra dependencies for llm serving api.

```bash
pip install --quiet \
  uvicorn \
  starlette \
  fastapi \
  sse_starlette \
  starlette_context
```

Run the model on cpu.

```bash
python3 -m llama_cpp.server \
  --model=Llama-3.2-3B-Instruct-Q8_0.gguf \
  --n_gpu_layers=-1 \
  --n_batch=512 \
  --n_ctx=4096 \
  --offload_kqv=False \
  --chat_format=llama-3 \
  --port=8080
```

Output:

```bash
load_tensors: loading model tensors, this can take a while... (mmap = true)
load_tensors: layer   0 assigned to device CPU
load_tensors: layer   1 assigned to device CPU
load_tensors: layer   2 assigned to device CPU
load_tensors: layer   3 assigned to device CPU
load_tensors: layer   4 assigned to device CPU
load_tensors: layer   5 assigned to device CPU
load_tensors: layer   6 assigned to device CPU
load_tensors: layer   7 assigned to device CPU
load_tensors: layer   8 assigned to device CPU
load_tensors: layer   9 assigned to device CPU
load_tensors: layer  10 assigned to device CPU
load_tensors: layer  11 assigned to device CPU
load_tensors: layer  12 assigned to device CPU
load_tensors: layer  13 assigned to device CPU
load_tensors: layer  14 assigned to device CPU
load_tensors: layer  15 assigned to device CPU
load_tensors: layer  16 assigned to device CPU
load_tensors: layer  17 assigned to device CPU
load_tensors: layer  18 assigned to device CPU
load_tensors: layer  19 assigned to device CPU
load_tensors: layer  20 assigned to device CPU
load_tensors: layer  21 assigned to device CPU
load_tensors: layer  22 assigned to device CPU
load_tensors: layer  23 assigned to device CPU
load_tensors: layer  24 assigned to device CPU
load_tensors: layer  25 assigned to device CPU
load_tensors: layer  26 assigned to device CPU
load_tensors: layer  27 assigned to device CPU
load_tensors: layer  28 assigned to device CPU
load_tensors: tensor 'token_embd.weight' (q8_0) (and 282 others) cannot be used with preferred buffer type CPU_AARCH64, using CPU instead
load_tensors:   CPU_Mapped model buffer size =  3255.90 MiB
warning: failed to mlock 426467328-byte buffer (after previously locking 0 bytes): Cannot allocate memory
Try increasing RLIMIT_MEMLOCK ('ulimit -l' as root).
```

The app loads embeddings onto the CPU as well.

```python
    model_kwargs = {'device':'cpu'}
    embeddings = HuggingFaceEmbeddings(model_kwargs=model_kwargs)
```

Run chatbot

```bash
chainlit run app.py -w --port 8081 --host 0.0.0.0
```

Performance (this takes ~4sec on a GPU) so x10times slower...

```bash
llama_perf_context_print:        load time =    8370.56 ms
llama_perf_context_print: prompt eval time =   14790.13 ms /   863 tokens (   17.14 ms per token,    58.35 tokens per second)
llama_perf_context_print:        eval time =   32913.92 ms /   233 runs   (  141.26 ms per token,     7.08 tokens per second)
llama_perf_context_print:       total time =   48512.33 ms /  1096 tokens
```
