import streamlit as st
from langchain.llms import VertexAI
# from langchain import PromptTemplate, LLMChain
# from langchain.document_loaders.csv_loader import CSVLoader
# from langchain.document_loaders import UnstructuredHTMLLoader
# from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceHubEmbeddings, LlamaCppEmbeddings
from langchain.chains import RetrievalQA

template = """
        You are an Expert on Kautilya's Arthashastra.  Give accurate answer to the following question.
        Under no circumstances do you give any answer outside of Arthashastra.
        
        ### QUESTION
        {question}
        ### END OF QUESTION
        
        Answer:
        """

st.title('Ask Kautilya')

#
# def generate_response(question):
#     prompt = PromptTemplate(template=template, input_variables=["question"])
#     llm = VertexAI()
#     llm_chain = LLMChain(prompt=prompt, llm=llm)
#     response = llm_chain.run({'question': question})
#     st.info(response)


def build_QnA_db():

    # loader = PyPDFLoader("./data/Arthashastra_of_Chanakya_English.pdf")
    loader = TextLoader("./data/Arthashastra_of_Chanakya_English.txt", encoding = 'UTF-8')

    docs = loader.load_and_split()
    print(docs)
    embeddings = HuggingFaceHubEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    retriver = db.as_retriever()
    llm = VertexAI()
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriver, verbose=False, chain_type="stuff")
    return chain


if "chain" not in st.session_state:
    st.session_state["chain"] = build_QnA_db()


def generate_response_from_db(question):
    chain = st.session_state["chain"]
    response = chain.run(question)
    st.info(response)


with st.form('my_form'):
    text = st.text_area('Ask Question:', '... about ArthaShastra')
    submitted = st.form_submit_button('Submit')
    if submitted:
        generate_response_from_db(text)
