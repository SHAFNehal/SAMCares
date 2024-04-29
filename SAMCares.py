'''
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

# Set Code to access other codes
import sys
# Path to code files
sys.path.append(r'./main_codes')

# Log in to huggingface for LLaMa2 Model & Weight Access
from huggingface_hub import login
login("hf_YOUR_HUGGINGFACE_TOKEN")


# # Import necessary files
import chainlit as cl
from llama2_model_text_generator import RAG_chat_bot
from database_preparation import create_vector_database

New_Data = int(input("Do you want to select a new data file? (1 = Yes and 0 = No):"))

# Decide to work on a new file or not
if New_Data == 1:
    create_vector_database()
else:
    pass

# chainlit code
@cl.on_chat_start
async def start():
    chain = RAG_chat_bot()
    msg = cl.Message(content="Starting SAMCares...")
    await msg.send()
    msg.content = "Hi, Welcome to SAMCares. What do you want to ask?"
    await msg.update()
    cl.user_session.set("chain", chain)
    

@cl.on_message
async def on_message(message: cl.Message):
    chat_history = cl.user_session.get("chat_history", [])
    chat_history.append(message.content)
    cl.user_session.set("chat_history", chat_history)
    chain = cl.user_session.get("chain")  # type: LLMChain
    res = await chain.arun(
                            query=message.content, 
                            callbacks=[cl.AsyncLangchainCallbackHandler(stream_final_answer=True)], 
                            stream_output=True,
                            chat_history=chat_history
                            )

    await cl.Message(content=res).send()


# chainlit run SAMCares.py -w
