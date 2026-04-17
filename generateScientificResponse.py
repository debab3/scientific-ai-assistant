from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from apis.openai_api import queryOpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from dotenv import load_dotenv

load_dotenv()
showDebugStatements = False


"""
Metrics
1) Relevant Statement: This ratio measures the fraction of relevant statements in the answer text in relation to the total number 
of statements.
2) Uncited Sources: This ratio metric measures the fraction of sources that are cited in the answer text in relation to the total 
number of listed sources.
3) Unsupported Statements: This ratio metric measures the fraction of relevant statements that are not factually supported by any 
of the listed sources. Any row of the factual support matrix with no checked cell corresponds to an unsupported statement.
4) Citation Accuracy: This ratio metric measures the fraction of statement citations that accurately reflect that a source’s 
content supports the statement. This metric can be computed by measuring the overlap between the citation and the factual support 
matrices, and dividing by the number of citations
5) Citation Thoroughness: This ratio metric measures the fraction of accurate citations included in the answer text compared to 
all possible accurate citations
"""


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
        if not author or author == ['']:
            author = "Unknown Author"
        elif len(author) <= 2:
            author = " &".join(author)
        else:
            author = author[0] + " et al."
        title = doc.metadata.get('title', 'Unknown Title')
        page = doc.metadata.get('page', 'Unknown Page')
        content = doc.page_content
        context_list.append(f"Source: {source}\nPage: {page}\nContent:{content}")
        context_info.append({
            "source": str(source),
            "author": str(author),
            "title": str(title),
            "page": str(page),
            "content": str(content)
        })
        
    consolidated_info = []
    for i in range(len(context_info)):
        if len(consolidated_info) == 0:
            consolidated_info.append(context_info[i])
        else:
            found = False
            for j in range(len(consolidated_info)):
                if context_info[i]['source'] == consolidated_info[j]['source'] and context_info[i]['page'] == consolidated_info[j]['page']:
                    consolidated_info[j]['content'] += ". " + context_info[i]['content']
                    found = True
                    break
            if found == False:
                consolidated_info.append(context_info[i])


    context_text = "\n\n".join(context_list)
    user_input = f"CONTEXT:\n{context_text}\n\nQUESTION: {query}"
    system_prompt = """
You are a highly rigorous Scientific Research Assistant. 
Your goal is to give a comprehensive answer to the user's question using the provided context snippets.
You must cite each claim you make with the format [Source, Page].
Do not list bullet points, make your claims as complete senteces with citations.
If you cannot answer the question using the context, only return \"I don't have enough information to answer this sorry\".
"""

    response = queryOpenAI(prompt = user_input, sysRole = system_prompt)
    return consolidated_info, response


def extractStatements(response):
    system_prompt = """
You are a Statement Consolidation Assistant.
Your goal is to analyse a response given below that has citations and consolidate each of the statements with their corresponding citations.
You must return ONLY a python list containing the statements as elements with their citations.
"""
    user_input = f"Response: {response}"
    consolidatedList = queryOpenAI(prompt=user_input, sysRole=system_prompt)
    consolidatedList = consolidatedList.split("\n")
    statements = []
    for i in range(len(consolidatedList)):
        consolidatedList[i] = consolidatedList[i].replace("\"", "").replace("\'","").replace("\n", "").replace("```python", "").replace("```", "")
        consolidatedList[i] = consolidatedList[i].strip()
        if len(consolidatedList[i]) > 2:
            statements.append(consolidatedList[i])
    return statements

def redoStatement(statement, source):
    system_prompt = """
You are a rigorous Scientific Research Assistant.
A statement below does not correspond to the source it refers to.
Redo the statement according to what the source says and include the same citation in the same format [Source, Page].
You should only return one redid statement.
"""
    user_prompt = f"\nStatement: {statement}\nSource: {source}"

    redidStatement = queryOpenAI(prompt = user_prompt, sysRole= system_prompt)
    return redidStatement

def verifyStatements(sources, foundStatements):
    ##check if citation is valid
    model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-base')
    tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-base')
    features = tokenizer(sources, foundStatements, padding=True, truncation=True, return_tensors="pt")
    
    model.eval()
    with torch.no_grad():
        scores = model(**features).logits
        label_mapping = ['contradiction', 'entailment', 'neutral']
        labels = [label_mapping[score_max] for score_max in scores.argmax(dim=1)]
    return labels


def evaluateReponse(context, statements, response, depth = 0):
    relevant_statements = 0.0
    uncited_sources = 0.0
    unsupported_statements = 0.0
    citation_accuracy = 0.0
    citation_thoroughness = 0.0
    correctedResponse = ""
    incorrectStatements = {}

    citationsFound = 0
    foundStatements = []
    sources = []
    ##Evaluate every statement
    for statement in statements:
        ##check if statement has a citation, if there is identify it
        if statement != '':
            citationFound = False
            citation = ""
            citationStart = None
            citationEnd = None
            for i in range(len(statement)):
                if statement[i] == "[":
                    citationStart = i
                if statement[i] == "]" and citationStart != None:
                    citationEnd = i
                    citationFound = True
                    citationsFound += 1
                    citation = statement[citationStart+1:citationEnd]
                    break
            
            curSource = None
            curPage = None
            if citationFound:
                citation = citation.split(",")
                if len(citation) == 2:
                    for i in range(len(citation)):
                        citation[i] = citation[i].split(":")
                    curSource = citation[0][1].strip()
                    curPage = citation[1][1].strip()
                    if showDebugStatements:
                        print(f"Citation Found, Source: {curSource}, Page: {curPage}")
                    for j in range(len(context)):
                        if context[j]['source'] == curSource and context[j]['page'] == curPage:
                            foundStatements.append(statement)
                            sources.append(curSource)
                            break
                else:
                    citationsFound -= 1
                        
    if citationsFound == 0 or len(foundStatements) == 0:
        newStatements = extractStatements(response=response)
        depth += 1
        if depth == 3:
            return ['nil'], response, None
        if showDebugStatements:
            print(f"Redoing Evaluation due to issues, Depth: {depth}\n{newStatements}\n{foundStatements}\n")
        evaluateReponse(context=context, statements=newStatements, response=response, depth=depth)
    
    labels = verifyStatements(sources=sources, foundStatements=foundStatements)
    if showDebugStatements:
        print(f"Lables: {labels}")
    
    # Metric Calculation
    for i, label in enumerate(labels):
        currentStatement = foundStatements[i]
        currentSource = sources[i]

        if label == 'entailment' or label == 'neutral':
            relevant_statements += 1
        elif label == 'contradiction':
            unsupported_statements += 1
            incorrectStatements[currentStatement] = currentSource
    
    if relevant_statements == 0:
        unsupported_statements = 1.0
    else:
        unsupported_statements /= relevant_statements
    relevant_statements /= len(statements)
    
    citation_accuracy = (relevant_statements / len(foundStatements)) if foundStatements else 0
    uniqueSources = 0
    seenSources = []
    for i in range(len(sources)):
        if sources[i] not in seenSources:
            seenSources.append(sources[i])
            uniqueSources += 1
    citation_thoroughness = uniqueSources / len(context)

    for i in range(len(context)):
        if context[i]['content'] not in sources:
            uncited_sources += 1
    
    uncited_sources /= len(context)


    # Redo Logic
    for i in range(len(statements)):
        if statements[i] in incorrectStatements:
            if showDebugStatements:
                print("Correcting incorrect statement")
            curStatement = statements[i]
            curSource = incorrectStatements[curStatement]
            attempt = 0
            while attempt != 3:
                redidStatement = redoStatement(curStatement, curSource)
                label = verifyStatements([curSource], [redidStatement])
                if label[0] == 'entailment' or label[0] == 'neutral':
                    if showDebugStatements:
                        print("Incorrect statement corrected")
                        break
                    nextLine = redidStatement
                attempt += 1
                if attempt == 3:
                    nextLine = curStatement
                    if showDebugStatements:
                        print("Couldn't correct incorrect statement")
        else:
            nextLine = statements[i]
        if correctedResponse == "":
            correctedResponse += nextLine
        else:
            correctedResponse += "\n" + nextLine

    stats = {
        "relevant": relevant_statements,
        "uncited": uncited_sources,
        "unsupported": unsupported_statements,
        "citationAcc": citation_accuracy,
        "citationTh": citation_thoroughness
    }
    return incorrectStatements, correctedResponse, stats


def getScientificResponse(query):
    context, response = generateResponse(query)
    if showDebugStatements:
        print("Generated Base Response")
    
    if response == "I don't have enough information to answer this sorry":
        return response, "Statements changed:\n\nNothing to change", {}
    
    ## Split by statement
    statements = extractStatements(response=response)
    if showDebugStatements:
        print(f"Extracted Statements\n{statements}")

    ##Evaluate Response
    incorrectStatements, correctedResponse, stats = evaluateReponse(context, statements, response)
    if showDebugStatements:
        print("Evaluated Response")

    changesToDisplay = "Statements changed:\n"
    if len(incorrectStatements) > 0:
        changesToDisplay += "\n".join(incorrectStatements)
    else:
        changesToDisplay += "\nNothing to change" 

    return correctedResponse, changesToDisplay, stats

if __name__ == "__main__":
    getScientificResponse(query="How do LLMs work?")