from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from apis.openai_api import queryOpenAI

def generateResponse(query):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = PineconeVectorStore(
        index_name="science-assistant-index", 
        embedding=embeddings,
        text_key="text" # This matches the key where your page content is stored
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.invoke(query)

    context_list = []
    context_info = []
    for doc in relevant_docs:
        source = doc.metadata.get('arxiv_id')
        author = doc.metadata.get('author', '')
        author = author.split(";")
        if author == '':
            print(True)
        elif len(author) <= 2:
            author = " &".join(author)
        else:
            author = author[0] + " et al."
        title = doc.metadata.get('title', 'Unknown Title')
        page = doc.metadata.get('page', 'Unknown Page')
        content = doc.page_content
        context_list.append(f"Source: {source}\nPage: {page}\nContent:{content}")
        context_info.append({
            "source": source,
            "author": author,
            "title": title,
            "page": page,
            "content": content
        })
    
    context_text = "\n\n".join(context_list)
    user_input = f"CONTEXT:\n{context_text}\n\nQUESTION: {query}"
    system_prompt = """
You are a highly rigorous Scientific Research Assistant. 
Your goal is to give a comprehensive answer to the user's question using the provided context snippets.
You must cite each claim you make with the format [Source, Page].
"""

    response = queryOpenAI(prompt = user_input, sysRole = system_prompt)
    return context_info, response


def redoStatement(statement):
    pass


def evaluateReponse(response):
    relevant_statements = 0.0
    uncited_sources = 0.0
    unsupported_statements = 0.0
    citation_accuracy = 0.0
    citation_thoroughness = 0.0
    pass


def getScientificResponse():
    pass

