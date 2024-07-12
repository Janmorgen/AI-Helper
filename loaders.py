import os
from langchain_community.vectorstores import Chroma
import uuid
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import TextLoader,DirectoryLoader,PyPDFDirectoryLoader,PyMuPDFLoader,PyPDFLoader, UnstructuredHTMLLoader,BSHTMLLoader

from langchain.docstore.document import Document
from tqdm import tqdm as tqdm
import nest_asyncio
from bs4 import BeautifulSoup
import urllib.request
from langchain_community.document_loaders import WebBaseLoader
import re
import jsonpickle
import json
class Loader:
    collectionStore="./vectorstore/documents/collections.json"
    text_splitter=CharacterTextSplitter(        
            separator = ".",
            chunk_size = 400,
                chunk_overlap  = 100, #striding over the text
            length_function = len,
        )
    def __init__(self,embedding,parent_dir,text_splitter=text_splitter):
        self.parent_dir=parent_dir
        self.text_splitter=text_splitter
        self.embedding=embedding
        self.curr_files = set([file for file in os.listdir(f"./{self.parent_dir}") if f".{self.parent_dir}" in file])
        print(f"Current files: {self.curr_files}")
        self.new_Docs=self.new_docs()
        print(f"New files: {self.new_Docs}")
        if self.parent_dir=="pdf":
            self.loader=PyPDFLoader
        else:
            self.loader=BSHTMLLoader
    def wrap_path(self, file):
        return  f"./{self.parent_dir}/{file}"
    def name_item(self,item):
        source=item.replace(".","").replace("/","").replace(" ","").replace("!","").replace(":","")[:50]
    def new_docs(self) -> set:
        if os.path.exists(self.collectionStore):
            loaders=[]
            with open(self.collectionStore) as f :
                loaders = jsonpickle.decode(json.load(f))    
            for i in loaders:
                if i["loader"].parent_dir == self.parent_dir:
                    return (self.curr_files.difference(i["loader"].curr_files))
        return (self.curr_files)

    def split_docs(self):
        # I want to do the document source seperation here to avoid multiple vectorstores for sublinks
        docs = None
        print(f"New Docs to split {self.new_Docs}")
        for i in self.new_Docs:
            filepath=self.wrap_path(i)
            print(f"Loading {filepath}...")
            loader = self.loader(filepath)
            if docs == None:
                docs = loader.load()
            else:
                docs.extend(loader.load())

        if docs!= None and len(docs)>0 and len(self.new_Docs)>0:
            if self.text_splitter!=None:
                all_splits = self.text_splitter.split_documents(docs)
                for i in all_splits:
                    if i.page_content==None:
                        all_splits.remove(i)

                print(f"Split into {len(all_splits)} chunks")

                return all_splits
            return docs
        else:
            return None

class PDFLoader(Loader):
    
    def __init__(self,embedding,parent_dir,text_splitter=Loader.text_splitter):
        super().__init__(embedding,parent_dir,text_splitter)
    



class WebPageLoader(Loader):
    def __init__(self, embedding, parent_dir,urls, text_splitter=Loader.text_splitter):
        super().__init__(embedding, parent_dir, text_splitter)
        self.loader=WebBaseLoader
        self.loader.requests_per_second=5
        self.urls = urls
        self.curr_files=self.get_sites()
        self.new_Docs=self.new_docs()
        
    def get_sites(self):
        sites=set()
        for url in self.urls:
            sites=sites.union(set(self.getChildUrls(url)))
        return sites
    def wrap_path(self, file):
        return file
    def split_docs(self):
        splitDocs= super().split_docs()
        if splitDocs == None:
            return None
        for i in splitDocs:
            string=""
            for l in i.page_content.split("\n"):
                
                if  "☰" in l:
                    continue
                else:
                    string=string+"\n"+l
            i.page_content=string
        return splitDocs
    def getChildUrls(self,url,exploreAdditional=None):
        p=re.compile('\/([\d]+)$')
        matchedResultNumbers=[]
        nest_asyncio.apply()

        parser = 'lxml'  # or 'lxml' (preferred) or 'html5lib', if installed

        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9"}
        http="https://"
        try:
            req = urllib.request.Request(http+url, headers=headers)
            resp = urllib.request.urlopen(req)
            soup = BeautifulSoup(resp, parser, from_encoding=resp.info().get_param('charset'))
            
            urls=[url]
            for link in soup.find_all('a', href=True):
                foundLink = link['href']
                if ('http' not in foundLink and urls[0]+foundLink not in urls and "#" not in foundLink):
                    regexSearch=p.match(foundLink)
                    if (regexSearch!=None and exploreAdditional!=None):
                        print(regexSearch.group(1))
                        matchedResultNumbers.append(int(regexSearch.group(1)))
                    else:
                        if foundLink[0]!="/":
                            foundLink = http+urls[0].split("/")[0]+"/"+foundLink
                        else:
                            foundLink = http+urls[0].split("/")[0]+foundLink
                        urls.append(foundLink)
        #             data = loader.load()
            urls[0]=http+urls[0]
            # if exploreAdditional!=None:
                # for i in range(min,max):
                    

        #             data = loader.load()
            return urls
        except Exception as e:
            print(f"Error: {e}")
    # def split_docs(self):
        
    #     docs = None
    #     for i in self.curr_files:
    #         if i not in files:
    #             files.add(i)
    #             filepath=i
    #             print(f"Loading {filepath}...")
    #             loader = self.loader(filepath)
    #             if docs == None:
    #                 docs = loader.aload()
    #             else:
    #                 docs.extend(loader.aload())


    #     if docs!= None and len(docs)>0:
    #         all_splits = self.text_splitter.split_documents(docs)
    #         for i in all_splits:
    #             if i.page_content==None:
    #                 all_splits.remove(i)

    #         print(f"Split into {len(all_splits)} chunks")

    #         with open(f"./{self.parent_dir}/{self.processed_file}",'w') as f:
    #             for u in files:
    #                 f.write(u+"\n")

    #         self.splits=all_splits
    #         return self.splits
    #     else:
    #         self.splits=None
    #         return self.splits

