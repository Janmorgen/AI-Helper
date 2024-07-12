from langchain_core.prompts import PromptTemplate, BasePromptTemplate, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.documents.base import *
from langchain.text_splitter import CharacterTextSplitter, HTMLHeaderTextSplitter
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings, 
)
import re
class Response:
    def __init__(self,query,response, documents, llm): 
        self.query=query
        self.response=response
        self.documents=documents
        self.llm=llm
        
        
    def format_sources(self):
        for source in self.documents:
        # print(source)
        # print(source.page_content)
            print(source.metadata['source']+"\n------------")


class Critic:
    def __init__(self,response,embedding_llm,llm,directory="./vectorstore/critiques"):
        self.response=response
        self.embedding_func=SentenceTransformerEmbeddings(model_name=embedding_llm)
        self.llm=llm
        self.directory= directory
        self.crit={}
        self.content=None
        self.context={}
        self.vectorstore=Chroma(persist_directory=self.directory,embedding_function=self.embedding_func)
    def critique(self):
        doc_prompt = ChatPromptTemplate.from_messages([("system",
"""
Document Contents: 
{page_content}

Document source: {source} 
""")])
        prompt = ChatPromptTemplate.from_messages(
            [("system", 
              """"

[INST]
You are a grader responsible for grading and providing critques to responses designed to answer queries.
You must grade the responses according to the following grade range rubric:
70+ pts for providing an execellent answer
40+ pts for answering the query fully and meeting query requirements
30+ pts for adhering to the information provided to responder
20+ pts for providing a focused and satisfactory response
10+ pts for providing a cohesive response
Only award higher grades if they meet the previous criteria as well


You must respond in the following JSON format, all fields must be included: 
``` JSON
"grade": integer <- grade you give the response
"critique": String <- a brief explanation of the grade you gave the response, include some areas for improvement if relevant
"successful": "true" OR "false"; whether the response satisfies the query fully
```
[/INST]
{context}
Qeury    : {query}
Response : {response}
              """
              ),
            #   ("user",
#                  """
# {query}
# """),("assistant","""
# Provided Documents:

# """),("ai",
#                  """

# Word count:
# {words}
# Page count:
# {pages}
# """)
])
        qa_chain=create_stuff_documents_chain(llm=self.llm,prompt=prompt,document_prompt=doc_prompt)
        prev_crits=self.vectorstore.as_retriever().get_relevant_documents(self.response.query)
        critique = qa_chain.invoke({"query":self.response.query, "response":self.response.response, "context":"","words":len(self.response.response.split()),"pages":len(self.response.response.split())/500})
        # print(critique)
        grade=re.search("\"grade\": ([0-9]+)",critique)
        crit =re.search("\"critique\": \"([\s\S]+)\",",critique)
        successful = re.search("\"successful\":[\s\"]+([a-z]+)[\s\"]{0}",critique)
        if (grade == None or crit==None or successful==None):
            print (f"Grade: {grade} -- Crit: {crit} -- Successful: {successful}\n")
            print(f"Parsing Failed: \n {critique}")
            return False
        else:
            self.crit["grade"]=grade.group(1)
            self.crit["critique"]=crit.group(1)
            self.crit["successful"]=successful.group(1)
            self.content=f"""
Previous Response
Query    : {self.response.query}

Response : {self.response.response}

Critique : {self.crit["critique"]}

            """

            self.context["source"]=""
        
    def print(self):
        print(
            f"""
Query    : {self.response.query}
Critique : 
    Satisfactory : {self.crit["successful"]}
    Rating : {self.crit["grade"]}/100
    Reason : {self.crit["critique"]}

            """)
    def add(self):
        doc = [Document(page_content=self.content,metadata=self.context)]
        # docs = CharacterTextSplitter("\n",chunk_size = 400,
        #             chunk_overlap  = 100, #striding over the text
        #         length_function = len,).split_documents(doc)
        # # print(doc)
        self.vectorstore.add_documents(doc)