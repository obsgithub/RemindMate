from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_ollama import ChatOllama
from typing import Union
import os

# Placeholder for the API key to protect privacy
os.environ["DASHSCOPE_API_KEY"] = "YOUR_DASHSCOPE_API_KEY_HERE"

class Communicate(BaseModel):
    """The conversation with the user is not yet over. What you say now is to communicate with the user and get their response."""
    communicate: str

class Task(BaseModel):
    """'task' is the description of the task to be executed by the task executor. 'communicate' is the communication content for the user."""
    task: str

class End(BaseModel):
    """End all operations. This is the final generated reply to the user."""
    end: str

class Act(BaseModel):
    """The action that should be executed."""
    action: Union[Communicate, Task, End] = Field(
        description="The action that should be executed. If you think you need to reply to the user, use Communicate. "
        "If you need to execute a task through the task executor, use Task. "
        "If the user's intent has been completed and no further communication is needed, use End."
    )

# Initialize the ChatOllama model instance
llm = ChatOllama(
    model="YOUR_MODEL_NAME_HERE",  # e.g., "qwen2.5:72b"
    temperature=0,
    base_url="http://YOUR_OLLAMA_HOST_IP:PORT"  # Placeholder for privacy
)

# Alternative model initialization (Commented out for demonstration)
# from langchain_community.chat_models.tongyi import ChatTongyi
# llm = ChatTongyi(model_name="YOUR_TONGYI_MODEL_NAME")
# a = llm.invoke("Hello")

if __name__ == "__main__":
    # Example usage for demonstration purposes only.
    # Requires valid API keys and a running Ollama service to execute.
    
    # abc = llm.with_structured_output(Task)
    # aa = abc.invoke("What should I do to open the refrigerator? Please output strictly according to the format.")
    # print(aa)
    
    pass