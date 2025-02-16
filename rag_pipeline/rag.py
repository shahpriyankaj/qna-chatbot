from langchain.vectorstores import FAISS
from utils import embeddings
from llama_cpp import Llama
from langchain import PromptTemplate
import json


class RAG:
    def __init__(self):
        #Load Faiss index (vector store)
        self.embedder = embeddings.Embeddings()
        # since the index was created by me, hence, I am using allow_dangerous_deserialization=True. Otherwise, the other options are to store index as JSON or use persistent storage option like Chroma Db or Redis or any other vector db.
        self.vectorstore = FAISS.load_local(folder_path = "../model", 
                                       embeddings = self.embedder.embeddings, 
                                       index_name= "faiss_vectorstore", 
                                       allow_dangerous_deserialization=True)

        #Load the llama model
        self.llm = Llama(
                model_path="model/llama-2-7b-chat.Q6_K.gguf",
                n_threads=16,  # For multi-threading on CPU cores
                n_batch=256,   # for faster inference
                n_ctx=1024     # Setting Context size 
                )     
        
        # System message for LLM prompt
        self.system_message = '''
                You are a helpful assistant for question-answering tasks. Provide an answer concisely based on the context only, without using general knowledge.
                Please correct any grammatical errors for improved readability. 
                If the context does not contain relevant information to answer the question, state that the answer is not available in the given context.
                '''
    
    def get_embeddings_and_prompt(self, query):
        # Retrieve relevant context
        question_embedding = self.embedder.embed_query(query)
        docs = self.vectorstore.similarity_search_by_vector(question_embedding, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        #Generate Prompt
        prompt = '''
                [INST] <<SYS>>{system_message}<</SYS>>
                Context: {context}
                Question: {question} [/INST]
                '''
        # Create a prompt template
        prompt_template = PromptTemplate(input_variables=["system_message", "context", "question"], template=prompt)

        # Format the final prompt with the query and search results
        final_prompt = prompt_template.format(system_message = self.system_message, question=query, context=context[:1024])
        return context, final_prompt
    

    def generate_response_for_evaluation(self, query):
        context, final_prompt = self.get_embeddings_and_prompt(query)
        response = self.llm(
                prompt=final_prompt,
                max_tokens=4000,
                temperature=0.5,
                top_p=0.9,
                repeat_penalty=1.2,
                top_k=20,
                echo=False
            )
        return context, response["choices"][0]["text"]
    

    def generate_response(self, query):
        context, final_prompt = self.get_embeddings_and_prompt(query)
        #Generate response using LLM and stream the output as it generates.
        for chunk in self.llm(
                prompt=final_prompt,
                max_tokens=4000,
                temperature=0.5,
                top_p=0.9,
                repeat_penalty=1.2,
                top_k=20,
                echo=False,
                stream=True
            ):
            json_chunk  = json.dumps({"response": chunk["choices"][0]["text"]})
            yield f"{json_chunk}\n"  # Send as newline-separated JSON