#https://github.com/sifubro/langchain-course/blob/main/chapters/04-chat-memory.ipynb




# 1. ConversationBufferMemory with RunnableWithMessageHistory

from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    ChatPromptTemplate
)

system_prompt = "You are a helpful assistant called Zeta."

prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{query}"),
])


pipeline = prompt_template | llm


from langchain_core.chat_history import InMemoryChatMessageHistory

chat_map = {}
def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chat_map:
        # if session ID doesn't exist, create a new chat history
        chat_map[session_id] = InMemoryChatMessageHistory()
    return chat_map[session_id]



from langchain_core.runnables.history import RunnableWithMessageHistory

pipeline_with_history = RunnableWithMessageHistory(
    pipeline,
    get_session_history=get_chat_history,
    input_messages_key="query",
    history_messages_key="history"
)

pipeline_with_history.invoke(
    {"query": "Hi, my name is James"},
    config={"session_id": "id_123"}
)


pipeline_with_history.invoke(
    {"query": "What is my name again?"},
    config={"session_id": "id_123"}
)



# 2. ConversationBufferWindowMemory with RunnableWithMessageHistory

from pydantic import BaseModel, Field
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage

class BufferWindowMessageHistory(BaseChatMessageHistory, BaseModel):
    messages: list[BaseMessage] = Field(default_factory=list)
    k: int = Field(default_factory=int)

    def __init__(self, k: int):
        super().__init__(k=k)
        print(f"Initializing BufferWindowMessageHistory with k={k}")

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add messages to the history, removing any messages beyond
        the last `k` messages.
        """
        self.messages.extend(messages)
        self.messages = self.messages[-self.k:]

    def clear(self) -> None:
        """Clear the history."""
        self.messages = []



chat_map = {}
def get_chat_history(session_id: str, k: int = 4) -> BufferWindowMessageHistory:
    print(f"get_chat_history called with session_id={session_id} and k={k}")
    if session_id not in chat_map:
        # if session ID doesn't exist, create a new chat history
        chat_map[session_id] = BufferWindowMessageHistory(k=k)
    # remove anything beyond the last
    return chat_map[session_id]




from langchain_core.runnables import ConfigurableFieldSpec

pipeline_with_history = RunnableWithMessageHistory(
    pipeline,
    get_session_history=get_chat_history,
    input_messages_key="query",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="session_id",
            annotation=str,
            name="Session ID",
            description="The session ID to use for the chat history",
            default="id_default",
        ),
        ConfigurableFieldSpec(
            id="k",
            annotation=int,
            name="k",
            description="The number of messages to keep in the history",
            default=4,
        )
    ]
)


pipeline_with_history.invoke(
    {"query": "Hi, my name is James"},
    config={"configurable": {"session_id": "id_k4", "k": 4}}
)



chat_map["id_k4"].clear()  # clear the history

# manually insert history
chat_map["id_k4"].add_user_message("Hi, my name is James")
chat_map["id_k4"].add_ai_message("I'm an AI model called Zeta.")
chat_map["id_k4"].add_user_message("I'm researching the different types of conversational memory.")
chat_map["id_k4"].add_ai_message("That's interesting, what are some examples?")
chat_map["id_k4"].add_user_message("I've been looking at ConversationBufferMemory and ConversationBufferWindowMemory.")
chat_map["id_k4"].add_ai_message("That's interesting, what's the difference?")
chat_map["id_k4"].add_user_message("Buffer memory just stores the entire conversation, right?")
chat_map["id_k4"].add_ai_message("That makes sense, what about ConversationBufferWindowMemory?")
chat_map["id_k4"].add_user_message("Buffer window memory stores the last k messages, dropping the rest.")
chat_map["id_k4"].add_ai_message("Very cool!")

chat_map["id_k4"].messages




pipeline_with_history.invoke(
    {"query": "what is my name again?"},
    config={"configurable": {"session_id": "id_k4", "k": 4}}
)

























