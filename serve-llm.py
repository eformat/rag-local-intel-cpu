from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp

llm = LlamaCpp(
    model_path="Llama-3.2-3B-Instruct-Q8_0.gguf",
    n_gpu_layers=-1,
    n_batch=512,
    n_ctx=4096,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)