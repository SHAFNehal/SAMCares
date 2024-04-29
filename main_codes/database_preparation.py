'''
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

#  This set of code will be used to generate the vector database depending on the selected file

# Import necessary data files
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, UnstructuredFileLoader, TextLoader

def load_user_data(data_types, directory):
    """
    Allows the user to select a data type and load a file from a specified directory.

    Parameters:
    data_types (list): List of data types (e.g., ['pdf', 'directory', 'unstructured', 'text']).
    directory (str): Path to the directory containing the files.
    """

    # Ensure the provided directory exists
    if not os.path.exists(directory):
        raise ValueError(f"Directory not found: {directory}")

    # List files in the directory
    files = os.listdir(directory)
    print("\nAvailable files:")
    for file in files:
        print(file)
        
    # Prompt user to select a data type
    print("Available data types:", data_types)
    print('Currently it only support PDF and TXT files!.\n')
    data_type = input("Enter the data type: ").lower()

    if data_type not in data_types:
        raise ValueError(f"Invalid data type selected: {data_type}")

    # Prompt user to select a file
    file_name = input("\nEnter the name of the file to load: ") + '.' + str(data_type)
    file_path = os.path.join(directory, file_name)

    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")

    # Load the file based on the selected data type
    if data_type == 'pdf':
        loader = PyPDFLoader(file_path)
    elif data_type == 'txt':
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported data type!")

    return loader.load()


def create_vector_database(data_types = ['pdf', 'txt'], 
                           directory = './data', 
                           vector_database_path = './vector_data',
                        #    vector_database_name = 'vector_database'
                           ):
    # Load the data from file
    # The function is going to interact with the user to learn 
    # 1. Which file type are they using and 
    # 2. Which file to open
    # We are limiting it to opening one file at a time for now
    documents = load_user_data(data_types, directory)

    # Split the Text and chunk it into pieces
    text_splitter=CharacterTextSplitter(separator='\n',
                                        chunk_size=512,#800
                                        chunk_overlap=200)

    text_chunks=text_splitter.split_documents(documents)

    # Load the Embedding model from sentence transformer
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cuda'})
    
    # Use the embedding to Create the vectore store for the database using the embedding
    vectorstore=FAISS.from_documents(text_chunks,
                                    embeddings)
    
    # Save the vector database into specific folder
    vectorstore.save_local(vector_database_path)

    print('#################################################################')
    print('# Completed creating vector database and saved in local folder! #')
    print('#################################################################')
