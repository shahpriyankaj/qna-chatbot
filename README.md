# qna-chatbot
QnA Chatbot using LLM and RAG



##Dont forget to add this- 
pip3 install huggingface-hub>=0.17.1
huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
or from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_K_M.gguf -> download the model

set CMAKE_ARGS=-DLLAMA_AVX2=ON

for more information about the Llama 2 7B Chat - GGUF models, please visit: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF


#If you face any issue installing llama-cpp-python, try following commands on cmd
pip install --upgrade pip setuptools wheel
pip install --upgrade scikit-build-core cmake ninja
pip install --no-cache-dir llama-cpp-python

#This worked without any issue
pip install llama-cpp-python==0.2.28


Advantages of FAISS
FAISS has various advantages, including:

Efficient similarity search: FAISS provides efficient methods for similarity search and grouping, which can handle large-scale, high-dimensional data.
GPU support: FAISS includes GPU support, which enables for further search acceleration and can greatly increase search performance on large-scale datasets.
Scalability: FAISS is designed to be extremely scalable and capable of handling large-scale datasets including billions of components.

![Alt Text](images/simple_rag_pipeline.png)
image source: https://medium.com/@drjulija/what-is-retrieval-augmented-generation-rag-938e4f6e03d1



3 Takeaways for the Research:

It uses graph transformer neural networks to generate high-dimensional embeddings for clinical vocabulary, enabling better integration of clinical knowledge into AI models.
The resource includes embeddings from seven medical vocabularies, offering a comprehensive view of clinical data.
The approach is hypothesis-free and doesn’t rely on patient-level information, making it scalable and applicable across diverse clinical environments.
Importance for Humana:

This research can support personalized medicine, enhance clinical decision-making, and improve outcomes by integrating rich, structured clinical knowledge into AI systems, potentially leading to more efficient healthcare delivery.
Including in Q&A Chatbot:

You could integrate these embeddings into the chatbot's backend to provide more accurate and context-aware medical responses, particularly for health-related queries involving complex clinical terms and concepts.