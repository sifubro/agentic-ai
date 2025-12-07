


## TO UNDERSTAND!!!
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def io_task(n):
    time.sleep(2)
    return n

def cpu_task(n):
    return sum(i*i for i in range(10_00000))

with ThreadPoolExecutor() as threads, ProcessPoolExecutor() as procs:
    f1 = threads.submit(io_task, 10)
    f2 = procs.submit(cpu_task, 10)

    print("IO:", f1.result())
    print("CPU:", f2.result())


'''
https://github.com/sifubro/ReAct-Agent-Implementation-from-Scratch-with-LangChain/blob/main/ReAct_agent_from_scratch.ipynb
'''











'''
Please explain what does stop_sequences do

Explain the arguments of a AIMEssage response from https://github.com/sifubro/langchain-course/blob/main/chapters/04-chat-memory.ipynb
AIMessage(content='Hi James! How can I assist you today?', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 11, 'prompt_tokens': 26, 'total_tokens': 37, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_72ed7ab54c', 'finish_reason': 'stop', 'logprobs': None}, id='run-1b91e1ed-387a-479e-beb2-7d7cb53bc0b7-0', usage_metadata={'input_tokens': 26, 'output_tokens': 11, 'total_tokens': 37, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
done, not_done = wait(futures, return_when=FIRST_COMPLETED)
'''