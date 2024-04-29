'''
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
#  This set of code will define the LLM model and run it as needed

# Import necessary data files

import torch

from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_llm_light():
    print('Loading light version of LLaMa2!')
    from langchain.llms import CTransformers
    # Load the locally downloaded model here
    llm = CTransformers(
                        model = "TheBloke/Llama-2-70B-Chat-GGUF",
                        model_file = "llama-2-7b-chat.Q4_K_M.gguf",
                        model_type="llama",
                        max_new_tokens=1024,
                        temperature=0.2,
                        )
    return llm

def load_llm():
    print('Loading heavy version of LLaMa2!')
    # Load the Tokenizer for the main model
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf") 
    # Load the Main model for Text Generation
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-chat-hf", 
                                                device_map='auto',
                                                torch_dtype=torch.bfloat16,
                                                use_auth_token=True,
                                                load_in_8bit=True,
                                                # load_in_4bit=True
                                                )
    # Define the huggingface Pipeline for text generation and interaction
    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer= tokenizer,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    max_new_tokens = 2048, # 1024
                    do_sample=True,
                    top_k=10,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    return_full_text=False, # <---- Comment OFF
                    # repetition_penalty=1.5
                    )
    # Define the final LLM module from HuggingFace
    llm=HuggingFacePipeline(
                            pipeline=pipe,
                            model_kwargs={'temperature':0.2} # To control the creativity. Make it 1 for full creative and 0 for maintaining the the context
                            )
    return llm

def set_custom_template():
    template =  """
                <<SYS>>
		        Youre name is "SAMCares". You are a friendly Study Buddy. 
		        Converse with users in a friendly manner. 
		        You were created as part of a project to see if AI LLMs can improve study experience for students.
		        You are a helpful, respectful and honest assistant.
		        Always answer as helpfully as possible, while being safe.  
		        Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
		        Please ensure that your responses are socially unbiased and positive in nature.
		        Don't answer more than what has been asked.
		        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
		        If you don't know the answer to a question, please don't share false information.
		        Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question.
		        Whenever the user says "Thanks" or "Thank you" be graceful and reply with a "Welcome!" to them. 
                <</SYS>>
                	Helpful Answer::
                [INST]
		        ------
		        <ctx>
		        {context}
		        </ctx>
		        ------
                <hs>
                	{history}
                </hs>
                ------
                {question}
                	Answer::
                [/INST]
                """
    qa_prompt = PromptTemplate(
                                input_variables=["history", 
                                                 "context", 
                                                 "question"],
                                template=template,
                                )
    
    return qa_prompt

def qa_chain(llm, vectorstore, qa_prompt):
    # Define the chain
    chain =  RetrievalQA.from_chain_type(llm=llm,
                                        chain_type = "stuff", # "stuff", "map_reduce", "map_rerank", and "refine".
                                        return_source_documents=False,
                                        retriever=vectorstore.as_retriever(search_type="similarity", 
                                                                           search_kwargs={"k": 6}),
                                        chain_type_kwargs={"verbose": False,
                                                            'prompt': qa_prompt,
                                                            "memory": ConversationBufferMemory(
                                                                                                memory_key="history",
                                                                                                input_key="question"
                                                                                                ),
                                                            }
                                        )
    return chain


def RAG_chat_bot(vector_database_path = './vector_data'):
    # Load the Embedding model from sentence transformer
    embeddings = HuggingFaceEmbeddings( model_name='sentence-transformers/all-MiniLM-L6-v2', # "sentence-transformers/all-mpnet-base-v2"  'sentence-transformers/all-MiniLM-L6-v2'
                                        model_kwargs={'device': 'cuda'})
    # Use the embedding from vectore store for the database using the embedding
    vectorstore = FAISS.load_local(vector_database_path, 
                                   embeddings,
                                   allow_dangerous_deserialization=True)
    # Load the LLM
    llm = load_llm()
    # llm = load_llm_light()
    # Get the qa_prmpt
    qa_prompt = set_custom_template()
    # Get the qa_Chain
    chain = qa_chain(llm, vectorstore, qa_prompt)

    return chain
