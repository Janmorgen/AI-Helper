from loaders import PDFLoader, WebPageLoader
from langchain.text_splitter import CharacterTextSplitter, HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor,LLMChainFilter,EmbeddingsFilter
from langchain.chains import LLMChain,SequentialChain, TransformChain
from langchain_community.vectorstores import Chroma
import chromadb
import os
from langchain_core.prompts import PromptTemplate, BasePromptTemplate, ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings, 
)
from langchain_community.embeddings import OllamaEmbeddings
import jsonpickle
import json
from langchain_core.documents.base import *
class Retriever:
    def __init__(self,embedding,llm):
        self.embedding=embedding
        self.llm=llm
        self.loaders=[] # list of dictionaries
        self.vectorstores=None
        self.collectionStore="./vectorstore/documents/collections.json"
    def initLoaders(self):
        pdfs = PDFLoader(self.embedding,"pdf")


        htmls = PDFLoader(self.embedding,"html", 
                        CharacterTextSplitter(
                            "\n",
                            chunk_size = 400,
                            chunk_overlap  = 100, #striding over the text
                            length_function = len,
                            )
                        )
        webResources= WebPageLoader(self.embedding,"code",
                                    [],# ["www.tweaked.cc/","www.minecraftitemids.com"], 
                                    CharacterTextSplitter(
                                        separator="title|\\n",
                                        is_separator_regex=True,
                                        chunk_size = 300,
                                        chunk_overlap  = 100, #striding over the text
                                        length_function=len
                                        )
                                    )
        # if all
        loaders=[pdfs,htmls,webResources]
        # loaders=[webResources]
        for i in range(len(loaders)):
            loader=loaders[i]
            # Structure
            # [i: 
            #    {
            #      "loader": loader[i]
            #      "docs"  : loader[i].split_docs()
            #    }
            #]
            if len(loader.curr_files.difference(loader.new_Docs))==0:
                self.loaders.append({"loader":loader})

    def load(self):
        for i in self.loaders:
            # Here is where metadata needs to be seperated if you eventually do it
            # This does not scan for new files rn if loaded from a save
            docs = i["loader"].split_docs()
            i["docs"]= docs

            # if docs!=None:
            #     self.docs.extend(docs)
    def createRetrievers(self):
        self.initLoaders()
        self.load()

        if self.loaders!=None and len(self.loaders)>0:
            for i in self.loaders:
                # print(SentenceTransformerEmbeddings(model_name=i["loader"].embedding))
                vectorstore=None
                if i["docs"]!=None and len(i["docs"])>0:
                    vectorstore = Chroma.from_documents(
                                                    persist_directory=f"./{i['loader'].parent_dir}/vectorstore",
                                                    documents=i["docs"], 
                                                    embedding=SentenceTransformerEmbeddings(model_name=i["loader"].embedding)
                                                    )
                else:
                    vectorstore = Chroma(
                                        persist_directory=f"./{i['loader'].parent_dir}/vectorstore",
                                        embedding_function=SentenceTransformerEmbeddings(model_name=i["loader"].embedding)
                                        )
                if (vectorstore!=None):
                    i["vectorstore"]=vectorstore
    def createRetrieversCollection(self):
        def documentSep(docs):
            documents={}
            for i in docs:
                source=i.metadata['source'].replace(".","").replace("/","").replace(" ","").replace("!","").replace(":","")[:50]
                if (source in documents.keys()):
                    #print("source found")
                    documents[source].append(i)
                else:
                    #print("initial")
                    documents[source]=[i]
            return documents
        self.initLoaders()
        self.load()
        if self.loaders!=None and len(self.loaders)>0:
            files=set()

            # Ensure that only docs that correspond to unique ids are kept and that only one of the duplicate ids is kept
            if os.path.exists(self.collectionStore):
                with open(self.collectionStore) as f:
                    lines=f.readlines()
                    for l in lines:
                        files.add(l.replace("\n",""))
            else:
                fp = open(self.collectionStore, 'x')
                fp.close()
            client = chromadb.PersistentClient(path="./vectorstore/documents")
            for i in self.loaders:
                i["vectorstore"]=[]
                if (i["docs"]!=None and len(i["docs"])>0):
                    seperatedDocs= documentSep(i["docs"])
                    for sourceKey in seperatedDocs.keys():
                        docs=seperatedDocs[sourceKey]
                        vectorstore=Chroma.from_documents(
                            client=client,
                            collection_name=sourceKey,
                            embedding=SentenceTransformerEmbeddings(model_name=i["loader"].embedding),
                            documents=docs
                            )
                        i["vectorstore"].append(vectorstore)
                        files.add(sourceKey)
                        

                    with open(self.collectionStore,'w') as f:
                        for u in files:
                            f.write(u+"\n")
                else:
                    for f in files:
                        vectorstore=Chroma(
                            client=client,
                            collection_name=f,
                            embedding_function=SentenceTransformerEmbeddings(model_name=i["loader"].embedding)
                        )
                        i["vectorstore"].append(vectorstore)
                    files=set()
                
    def createRetrieversCollectionJSON(self):

        def documentSep(docs):
            documents={}
            for i in docs:
                source=i.metadata['source'].replace(".","").replace("/","").replace(" ","").replace("!","").replace(":","")[:50]
                if (source in documents.keys()):
                    #print("source found")
                    documents[source].append(i)
                else:
                    #print("initial")
                    documents[source]=[i]
            return documents
        if os.path.exists(self.collectionStore):
            with open(self.collectionStore) as f :     
                self.loaders=jsonpickle.decode(json.load(f))
                print(f"Loaded data")
                print(self.loaders)
        
        self.initLoaders()
        self.load()
        testVar=False
        if self.loaders!=None and len(self.loaders)>0:
            client = chromadb.PersistentClient(path="./vectorstore/documents")
            newData=False
            for i in self.loaders:
                embeddingFunction = SentenceTransformerEmbeddings(model_name=i["loader"].embedding)
                print(i.keys())
                if ("sourceKey" not in i.keys()):
                    print (f"No SourceKey pipeline")
                    newData=True
                    i["vectorstore"]=[]
                    i["sourceKey"]=set()
                    if (i["docs"]!=None and len(i["docs"])>0):
                        print (f"Docs Found!")
                        seperatedDocs= documentSep(i["docs"])
                        for sourceKey in seperatedDocs.keys():
                            docs=seperatedDocs[sourceKey]
                            vectorstore=Chroma.from_documents(
                                client=client,
                                collection_name=sourceKey,
                                embedding=embeddingFunction,
                                documents=docs
                                )
                            i["vectorstore"].append(vectorstore)
                            i["sourceKey"].add(sourceKey)
                    # else:
                    #     for sourceKey in i["sourceKey"]:
                    #         vectorstore=Chroma(
                    #             client=client,
                    #             collection_name=sourceKey,
                    #             embedding_function=SentenceTransformerEmbeddings(model_name=i["loader"].embedding)
                    #         )
                    #         i["vectorstore"].append(vectorstore)
                else:
                    i["vectorstore"]=[]
                    for sourceKey in i["sourceKey"]:
                        vectorstore=Chroma(
                                client=client,
                                collection_name=sourceKey,
                                embedding_function=embeddingFunction,
                                )
                        i["vectorstore"].append(vectorstore)
            if (newData):
                vectorstores=[]
                for i in self.loaders:
                    i["docs"]=None
                    i["loader"].new_Docs=set()
                    vectorstores.append(i["vectorstore"])
                    i["vectorstore"]=None
                # The biggest impact on saving space could be not saving the embeddingfunction or the vectorstores
                with open(self.collectionStore,'w') as f:
                    json.dump(jsonpickle.encode(self.loaders),f)
                    print(f"Saved data")
                for i in range(len(vectorstores)):
                    self.loaders[i]["vectorstore"]=vectorstores[i]
    def createRetrieversJSON(self):
        def documentSep(docs):
            documents={}
            for i in docs:
                source=i.metadata['source'].replace(".","").replace("/","").replace(" ","").replace("!","").replace(":","")[:50]
                if (source in documents.keys()):
                    #print("source found")
                    documents[source].append(i)
                else:
                    #print("initial")
                    documents[source]=[i]
            return documents
        if os.path.exists(self.collectionStore):
            with open(self.collectionStore) as f :     
                self.loaders=jsonpickle.decode(json.load(f))
                print(f"Loaded data")
                print(self.loaders)
        
        self.initLoaders()
        self.load()
        testVar=False
        if self.loaders!=None and len(self.loaders)>0:
            client = chromadb.PersistentClient(path="./vectorstore/documents")
            newData=False
            for i in self.loaders:
                # embeddingFunction = SentenceTransformerEmbeddings(model_name=i["loader"].embedding)
                embeddingFunction = OllamaEmbeddings(model="llama3")
                if (i["docs"]!=None and len(i["docs"])>0):
                        print (f"Embedding new documents...")
                        newData=True
                        vectorstore=Chroma.from_documents(
                            client=client,
                            collection_name=i["loader"].parent_dir,
                            embedding=embeddingFunction,
                            documents=i["docs"]
                            )
                        i["vectorstore"]=vectorstore
                else:
                    vectorstore=Chroma(
                            client=client,
                            collection_name=i["loader"].parent_dir,
                            embedding_function=embeddingFunction,
                            )
                    i["vectorstore"]=vectorstore
            if (newData):
                vectorstores=[]
                for i in self.loaders:
                    i["docs"]=None
                    i["loader"].new_Docs=set()
                    vectorstores.append(i["vectorstore"])
                    i["vectorstore"]=None
                # The biggest impact on saving space could be not saving the embeddingfunction or the vectorstores
                with open(self.collectionStore,'w') as f:
                    json.dump(jsonpickle.encode(self.loaders),f)
                    print(f"Saved data")
                for i in range(len(vectorstores)):
                    self.loaders[i]["vectorstore"]=vectorstores[i]
    def FilterDocs(self, query, initK=5):
        if len(self.loaders)==0:
            self.createRetrievers()
        prompt = ChatPromptTemplate.from_messages(
            [("system", "[INST] Very Important : It is your responsibility to determine whether this document is relevant to answering the query, if it is not answer with only NO, if it is answer with only YES. Answer with only a \"YES\" or a \"NO\", do not respond with anything else.\n\n Query:\n{query}[/INST]\n\n Context:\nDocument Contents: \n{content}\n\nDocument source: \n{source} ")]
        )
        # Filter documents based on prompt
        embeddings_filter = LLMChainFilter.from_llm(llm=self.llm,prompt=prompt)
        docs=[]
        for i in self.loaders:
            # Generate multiple queries based on the original query and get more relevant documents
            retriever = MultiQueryRetriever.from_llm(
                retriever=i["vectorstore"].as_retriever(search_kwargs={"k":initK}), llm=self.llm
            )
            # Put both systems together and pull only documents which can answer the query
            retriever = ContextualCompressionRetriever(
                base_compressor=embeddings_filter, base_retriever=retriever,tags=["page_content","query"],_expects_other_args=True
            )
            docs.extend(retriever.invoke(query,config=RunnableConfig(tags=["page_content","query"],max_concurrency=10)))
        return docs
    def FilterDocsALT(self,query,initK=5):
        if len(self.loaders)==0:
            self.createRetrievers()
        prompt = ChatPromptTemplate.from_messages(
            [("system", "[INST] Very Important : It is your responsibility to determine whether this document is relevant to answering the query, if it is not answer with only NO, if it is answer with only YES. Answer with only a \"YES\" or a \"NO\", do not respond with anything else.\n\n Query:\n{query}[/INST]\n\n Context:\nDocument Contents: \n{content}\n\nDocument source: \n{source} ")]
        )
        # Filter documents based on prompt
        embeddings_filter = LLMChainFilter.from_llm(llm=self.llm,prompt=prompt)
        retrievers=[]
        docs=[]
        for i in self.loaders:

            retriever=i["vectorstore"].as_retriever(search_kwargs={"k":initK})
            retriever = ContextualCompressionRetriever(
                base_compressor=embeddings_filter, base_retriever=retriever,tags=["page_content","query"],_expects_other_args=True
            )
            retrievers.append(retriever)
        docs = EnsembleRetriever(retrievers=retrievers).get_relevant_documents(query)
        return docs
    
    def FilterDocsMaxRatio(self,query, initK = 10):
        def parser(text):
            import re
            p = re.compile('yes.',re.IGNORECASE)
            if (p.search(text)!=None):
                return True
            else:
                return False

        if len(self.loaders)==0:
            self.createRetrievers()
        prompt = ChatPromptTemplate.from_messages(
            [("system", "[INST] Decide whether the document is relavant to someone trying to effectively answer the query. Answer with only a \"YES\" or a \"NO\" do not provide any additional information relating to the documents or your thoughts on why documents are to be exlcuded or included[/INST] \n\n Query:\n{query}\n\n Context:\nDocument Contents: \n{content}\n\nDocument source: \n{source} ")]
        )

            
        embeddings_filter = LLMChain(llm=self.llm,prompt=prompt)
        def chainFilter(query, docs):
            output=[]
            for i in docs:
                response = embeddings_filter.invoke({"query":query,"content":i.page_content,"source":i.metadata["source"]})
                # print(response["text"])
                # print(parser(response["text"]))
                if (parser(response["text"])):
                    output.append(i)
            return output
        retrievers=[]
        docs=[]
        maxRatio=0
        for i in self.loaders:

            retriever=i["vectorstore"].as_retriever(search_kwargs={"k":initK})
            retriever = MultiQueryRetriever.from_llm(
                retriever=retriever, llm=self.llm
            )
            newDocs=chainFilter(query,retriever.get_relevant_documents(query))
            calc=(len(newDocs)/initK)
            print(calc)
            if (calc>0):
                maxRatio=calc
                retrievers.append(retriever)

        docs = EnsembleRetriever(retrievers=retrievers).get_relevant_documents(query,kwargs={"k":initK})[:initK]

        return docs
    
    def FilterDocsMaxRatioCol(self,query, initK = 10):
        def parser(text):
            import re
            p = re.compile('yes.',re.IGNORECASE)
            if (p.search(text)!=None):
                return True
            else:
                return False

        if len(self.loaders)==0:
            self.createRetrieversCollection()
        prompt = ChatPromptTemplate.from_messages(
            [("system", "[INST] Decide whether the document is relavant to someone trying to effectively answer the query. Answer with only a \"YES\" or a \"NO\" do not provide any additional information relating to the documents or your thoughts on why documents are to be exlcuded or included[/INST] \n\n Query:\n{query}\n\n Context:\nDocument Contents: \n{content}\n\nDocument source: \n{source} ")]
        )

            
        embeddings_filter = LLMChain(llm=self.llm,prompt=prompt)
        def chainFilter(query, docs):
            # print(docs)
            output=[]
            for i in docs:
                response = embeddings_filter.invoke({"query":query,"content":i.page_content,"source":i.metadata["source"]})
                # print(response["text"])
                # print(parser(response["text"]))
                print(".",end="")
                if (parser(response["text"])):
                    output.append(i)
            return output
        retrievers=[]
        docs=[]
        maxRatio=0
        for i in self.loaders:
            for vectorstore in i["vectorstore"]:
                retriever=vectorstore.as_retriever(search_kwargs={"k":initK})
                retriever = MultiQueryRetriever.from_llm(
                    retriever=retriever, llm=self.llm
                )
                newDocs=chainFilter(query,retriever.get_relevant_documents(query))
                calc=(len(newDocs)/initK)
                print(calc)
                if (calc>maxRatio):
                    maxRatio=calc
                    retrievers.append(retriever)

        docs = EnsembleRetriever(retrievers=retrievers).get_relevant_documents(query,kwargs={"k":initK})

        return docs
    def FilterDocsMaxRatioColTest(self,query, initK = 5):
        def parser(text):
            import re
            p = re.compile('yes',re.IGNORECASE)
            if (p.search(text)!=None):
                return True
            else:
                return False

        # if len(self.loaders)==0:
        #     self.createRetrieversCollection()
        summaryPrompt = ChatPromptTemplate.from_messages(
            [("system", "[INST] Write a summary of the provided context to the best of your ability, please adhere closely to the context and dont make assumptions from information found outside the content itself. [/INST] \n\nDocument Contents: \n{content}\nDocument concepts: \n{concepts}")]
        )
        filter = ChatPromptTemplate.from_messages(
            [("system", "[INST] Decide whether the context could be helpful to someone trying to effectively answer the query, please be lenient in your choice as you are trying to provide as much information as possible related to the query. The context does not need to explicitly answer the query to be useful. Answer with only a \"YES\" or a \"NO\".[/INST] \n\n Query:\n{query}\n\n Context:\nDocument Summary: \n{summary}\n\nDocument source: \n{source} ")]
        )
        conceptExtractor = ChatPromptTemplate.from_messages(
            [("system", "[INST] Decompose the subject matters explored by the document into different recognisable concepts, also provide a potential title for the document based on extracted concepts [/INST] \nContext:\nDocument Content: \n{content}\n\nDocument source: \n{source} ")]
        )
        factchecker = ChatPromptTemplate.from_messages(
            [("system", "[INST] You are a fact checker editor. Determine whether the response provides an accurate summary of the given document information, rewrite the summary with corrections that closely adhere to the provided document content[/INST] \n Response: {summary}\nDocument Content: \n{content}")]
        )
        contextadder = ChatPromptTemplate.from_messages(
            [("system", "[INST] Use the related concepts provided to you to add context to the query, please rewrite the question so that the relevant concepts are added to it[/INST] \n Question: {query}\nRelated concepts: \n{content}")]
        )
        # multiqueries = ChatPromptTemplate.from_messages(
        #     [("system", "[INST] Recontextualize the given query to match the documents provided [/INST] \nContext:\nPrevious concepts: {prevconcepts}\nDocument Content: \n{content}\n\nDocument source: \n{source} ")]
        # )

            
        embeddings_filter = LLMChain(llm=self.llm,prompt=filter)
        summary_filter = LLMChain(llm=self.llm,prompt=summaryPrompt)
        conceptextraction=LLMChain(llm=self.llm,prompt=conceptExtractor)
        factchecker=LLMChain(llm=self.llm,prompt=factchecker)
        contextadder=LLMChain(llm=self.llm,prompt=contextadder)
        # def transform_retrieval(inputs):
        #     docs=retriever.get_relevant_documents(query=inputs["question"])
        #     docs=[d.page_content for d in docs]
        #     docs_dict={
        #         "query":inputs["question"],
        #         "context": "\n---\n".join(docs)
        #     }
        #     return docs_dict
        # retrieval_chain=TransformChain(
        #     input_variables=["question"],
        #     output_variables=["query","contexts"],
        #     transform = transform_retrieval
        # )
        # rag_chain = SequentialChain(
        # chains=[retrieval_chain, embeddings_filter],
        # input_variables=["question"],  # we need to name differently to output "query"
        # output_variables=["query", "contexts", "text"]
# )
        def chainFilter(query, docs):
            # print(docs)
            output=[]
            summarizedDocs=[]
            client = chromadb.PersistentClient(path="./vectorstore/documents")
            embeddingFunction = OllamaEmbeddings(model="llama3")
            vectorstore=Chroma(
                            client=client,
                            collection_name="summaries",
                            embedding_function=embeddingFunction,
                            )
            # contextGrab=vectorstore.similarity_search_with_score(query)[0][0]

            # query=contextadder.invoke({"query":query,"content":contextGrab.page_content})["text"]
            print(f"New Query: {query}")
            for i in docs:
                existing_summary =vectorstore.get(where={"source":i.page_content})["documents"]
                print("\n\n-------------------------------------------------")
                print(existing_summary)
                if (len(existing_summary)==0):
                    concepts= conceptextraction.invoke({"content":i.page_content,"source":i.metadata["source"]})["text"]
                    print(f"Concepts: {concepts}")
                    
                    summary=summary_filter.invoke({"content":i.page_content,"concepts":concepts})["text"]
                    print(f"\nSummary: {summary}")
                    # fact_summary=factchecker.invoke({"summary":summary,"content":i.page_content})["text"]
                    # print(f"\nImproved Summary: {fact_summary}")
                    pageContent=f"{concepts}\n{summary}"
                    meta={"source":i.page_content}
                    summarizedDocs.append(Document(page_content=pageContent,metadata=meta))
                else:
                    summary=existing_summary[0]
                response = embeddings_filter.invoke({"query":query,"summary":summary,"source":i.metadata["source"]})["text"]

                
                print(f"\nFilter Response: {response}")
                # print(parser(response["text"]))
                
                if (parser(response)):
                    output.append(i)
            if(len(summarizedDocs)!=0):
                vectorstore.add_documents(summarizedDocs)
            return output

        docs=[]
        retrievers=[]
        client = chromadb.PersistentClient(path="./vectorstore/documents")

        for i in self.loaders:
            retriever=i["vectorstore"]
            # retriever = MultiQueryRetriever.from_llm(
            #     retriever=retriever, llm=self.llm
            # )
            for doc in retriever.similarity_search_with_score(query)[:10]:
                docs.append(doc[0])
            # newDocs=chainFilter(query,retriever.get_relevant_documents(query))
            # calc=(len(newDocs)/initK)
            # print(calc)
            # if (calc>0):
            #     maxRatio=calc
            #     retrievers.append(retriever)
        print(docs)
        # docs=EnsembleRetriever(retrievers=retrievers).get_relevant_documents(query,kwargs={"k":initK})
        docs=chainFilter(query,docs)[:10]
        return docs

