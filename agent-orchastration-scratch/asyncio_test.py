import asyncio

## 1️⃣ Blocking example (standard file open) — BAD

async def blocking_reader():
    print("Blocking reader: start")
    with open("./bigfile.bin", "rb") as f:
        while chunk := f.read(4096):
            print("Blocking reader: read chunk")
            # NO await → event loop cannot switch tasks
    print("Blocking reader: end")

async def periodic():
    while True:
        print("Periodic: tick")
        await asyncio.sleep(0.01)

async def main_blocking():
    task1 = asyncio.create_task(blocking_reader())
    task2 = asyncio.create_task(periodic())
    await asyncio.gather(task1, task2)


## 2️⃣ Non-blocking example (aiofiles)

import asyncio, aiofiles

async def nonblocking_reader():
    print("Nonblocking reader: start")
    async with aiofiles.open("bigfile.bin", "rb") as f:
        while True:
            chunk = await f.read(4096)
            if not chunk:
                break
            print("Nonblocking reader: read chunk")
    print("Nonblocking reader: end")



async def nonblocking_reader_no_aiofiles():
    print("Nonblocking reader: start")
    async with open("bigfile.bin", "rb") as f:
        while True:
            chunk = await f.read(4096)
            if not chunk:
                break
            print("Nonblocking reader: read chunk")
    print("Nonblocking reader: end")


async def main_non_blocking():
    task1 = asyncio.create_task(nonblocking_reader())
    task2 = asyncio.create_task(periodic())
    tasks = [task1, task2]
    await asyncio.gather(*tasks)

    # await asyncio.gather(
    #     asyncio.create_task(nonblocking_reader()),
    #     asyncio.create_task(periodic()),
    # )


## 3️⃣ Non-blocking + cooperative yield
## Now cancellation is instant!!!

async def cancellable_reader():
    print("Reader: start")
    async with aiofiles.open(r"C:\Users\SiFuBrO\Desktop\SCRIPTS!!!!!\GitHub\agentic-ai\agent-orchastration-scratch\bigfile.bin", "rb") as f:
        while True:
            chunk = await f.read(4096)
            if not chunk:
                break
            print("Reader: read chunk")
            await asyncio.sleep(0)  # <- cooperative yield
    print("Reader: end")



async def main_non_blocking_cooperative_yield():
    task = asyncio.create_task(cancellable_reader())
    await asyncio.sleep(1)  # allow some chunks
    print("Main: cancelling reader")
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("Reader cancelled cleanly")




if __name__ == "__main__":

    # asyncio.run(main_blocking())
    # asyncio.run(main_non_blocking())
    # asyncio.run(nonblocking_reader_no_aiofiles())
    asyncio.run(main_non_blocking_cooperative_yield())

    # TODO:
    # 1. Process network request, I/O, call gpts, other tasks concurently
    # 2. Access a DB and write stuff to it
    # 
    # 





