import os
import getpass
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# Set up Mistral API key
if not os.environ.get("MISTRAL_API_KEY"):
    os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter your Mistral API key: ")

# Initialize AI components
print("Initializing AI components...")
llm = ChatMistralAI(model="mistral-large-latest")
embeddings = MistralAIEmbeddings(model="mistral-embed")
vector_store = InMemoryVectorStore(embeddings)

# Path to your tax documents folder - UPDATE THIS PATH
tax_docs_path = os.path.join(os.path.dirname(__file__), "TaxDocuments")

# Load documents from each country folder
all_documents = []
countries = ["USA", "Spain", "Portugal", "Germany", "France", "Netherlands", "Mexico"]

for country in countries:
    country_path = os.path.join(tax_docs_path, country)
    
    # Skip if folder doesn't exist
    if not os.path.exists(country_path):
        print(f"Warning: Folder for {country} not found at {country_path}")
        continue
    
    print(f"Loading documents from {country}...")
    
    # Load PDF documents
    loader = DirectoryLoader(country_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    
    print(f"Found {len(docs)} documents for {country}")
    
    # Add country metadata to each document
    for doc in docs:
        doc.metadata["country"] = country
        filename = os.path.basename(doc.metadata.get("source", ""))
        
        # Extract tax category from filename
        if "income" in filename.lower():
            doc.metadata["category"] = "Income Tax"
        elif "digital_nomad" in filename.lower():
            doc.metadata["category"] = "Digital Nomad Visa"
        elif "tax_treaty" in filename.lower():
            doc.metadata["category"] = "Tax Treaties"
        else:
            doc.metadata["category"] = "General"
        
        all_documents.append(doc)

print(f"Loaded {len(all_documents)} tax documents total")

# Split documents into chunks
print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(all_documents)
print(f"Split into {len(all_splits)} chunks")

# Store in vector database
print("Adding chunks to vector store...")
vector_store.add_documents(documents=all_splits)
print("Documents indexed successfully")

# Create custom prompt for tax analysis
tax_analysis_template = """You are a tax planning specialist for digital nomads. Use the following information to provide a detailed tax analysis for the user's situation.

User Profile:
- Citizenship: {citizenship}
- Target Country: {target_country}
- Stay Duration: {stay_duration} days per year
- Income Source: {income_source}
- Annual Income: {income_range}
- Business Structure: {business_structure}
- Special Tax Status Interest: {special_tax_interest}
- Home Office Use: {home_office}
- Long-term Plans: {long_term_plans}

Context from tax documentation:
{context}

Based on the user profile and the provided documentation, create a comprehensive tax analysis covering:
1. Residence and tax status implications
2. Applicable tax rates and obligations
3. Available tax benefits and special regimes
4. Recommended tax planning strategies
5. Key considerations and potential pitfalls

Be specific about which country's tax laws you are referring to. Provide accurate information based on the documentation.

Tax Analysis:"""

tax_analysis_prompt = PromptTemplate.from_template(tax_analysis_template)

# Define the questionnaire
questions = [
    {
        "id": "citizenship",
        "question": "What is your citizenship status?",
        "type": "text"
    },
    {
        "id": "target_country",
        "question": "Which country are you considering for digital nomad opportunities?",
        "type": "choice",
        "options": ["USA", "Spain", "Portugal", "Germany", "France", "Netherlands", "Mexico", "Other (please specify)"]
    },
    {
        "id": "stay_duration",
        "question": "How many days per year do you plan to spend in your target country?",
        "type": "choice",
        "options": ["Less than 90 days", "90-183 days", "More than 183 days", "Variable/not sure"]
    },
    {
        "id": "income_source",
        "question": "What is your primary source of income?",
        "type": "choice",
        "options": ["Employment with a foreign company", "Self-employment/freelancing", "Business ownership", "Investment income", "Mixed sources"]
    },
    {
        "id": "income_range",
        "question": "What is your approximate annual income range (in EUR or USD)?",
        "type": "choice",
        "options": ["Below €30,000", "€30,000 - €60,000", "€60,000 - €100,000", "€100,000 - €200,000", "Above €200,000"]
    },
    {
        "id": "business_structure",
        "question": "What type of business structure do you currently use or plan to establish?",
        "type": "choice",
        "options": ["Sole proprietorship/individual entrepreneur", "Limited liability company (LLC or equivalent)", "Corporation", "Partnership", "Not applicable/employed by others"]
    },
    {
        "id": "special_tax_interest",
        "question": "Are you interested in applying for a special tax status (such as Non-Habitual Resident, Digital Nomad Visa, 30% Ruling, etc.)?",
        "type": "choice",
        "options": ["Yes", "No", "Need more information"]
    },
    {
        "id": "home_office",
        "question": "Do you work primarily from a home office and plan to claim related deductions?",
        "type": "choice",
        "options": ["Yes", "No", "Mixed arrangement"]
    },
    {
        "id": "long_term_plans",
        "question": "How long do you plan to stay in your target country?",
        "type": "choice",
        "options": ["Less than 1 year", "1-3 years", "3-5 years", "5+ years or considering permanent residency"]
    }
]

# Define state and functions for the RAG application
class State(TypedDict):
    profile: dict
    context: List[Document]
    analysis: str

def retrieve(state: State):
    # Extract country from the profile
    target_country = state["profile"]["target_country"]
    if "Other" in target_country:
        # Extract country name from "Other (please specify)" format
        parts = target_country.split("(")
        if len(parts) > 1:
            specified_country = parts[1].replace("please specify)", "").replace(")", "").strip()
            if specified_country:
                target_country = specified_country
    
    # Create a query based on the user profile
    query = f"Tax requirements for digital nomads in {target_country} with {state['profile']['income_source']} income staying {state['profile']['stay_duration']} per year"
    
    # Add special status to query if interested
    if state["profile"]["special_tax_interest"] == "Yes":
        query += f" special tax status digital nomad visa"
    
    print(f"Searching with query: {query}")
    
    # Retrieve relevant documents, prioritizing the target country
    country_docs = vector_store.similarity_search(
        query,
        filter=lambda doc: doc.metadata.get("country") == target_country,
        k=3
    )
    
    # Get additional general documents
    general_docs = vector_store.similarity_search(
        query,
        k=3
    )
    
    # Combine and deduplicate
    all_relevant_docs = country_docs
    doc_ids = {doc.metadata.get("source") for doc in country_docs}
    
    for doc in general_docs:
        if doc.metadata.get("source") not in doc_ids:
            all_relevant_docs.append(doc)
            if len(all_relevant_docs) >= 5:
                break
    
    return {"context": all_relevant_docs}

def generate(state: State):
    # Format retrieved documents for the prompt
    docs_content = "\n\n".join([
        f"[Document from {doc.metadata.get('country')}, Category: {doc.metadata.get('category')}]\n{doc.page_content}"
        for doc in state["context"]
    ])
    
    # Generate the analysis
    messages = tax_analysis_prompt.invoke({
        "citizenship": state["profile"]["citizenship"],
        "target_country": state["profile"]["target_country"],
        "stay_duration": state["profile"]["stay_duration"],
        "income_source": state["profile"]["income_source"],
        "income_range": state["profile"]["income_range"],
        "business_structure": state["profile"]["business_structure"],
        "special_tax_interest": state["profile"]["special_tax_interest"],
        "home_office": state["profile"]["home_office"],
        "long_term_plans": state["profile"]["long_term_plans"],
        "context": docs_content
    })
    response = llm.invoke(messages)
    return {"analysis": response.content}

# Build and compile the graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Function to gather user responses through the questionnaire
def collect_user_responses():
    profile = {}
    print("\n" + "="*50)
    print("Digital Nomad Tax Planning Questionnaire")
    print("="*50)
    print("\nPlease answer the following questions to receive a personalized tax analysis.\n")
    
    for question in questions:
        print(f"\n{question['question']}")
        
        if question['type'] == 'choice':
            for i, option in enumerate(question['options'], 1):
                print(f"  {i}. {option}")
            
            while True:
                choice = input("\nEnter your choice (number): ")
                try:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(question['options']):
                        profile[question['id']] = question['options'][choice_idx]
                        break
                    else:
                        print("Invalid choice. Please try again.")
                except ValueError:
                    print("Please enter a number.")
        else:
            profile[question['id']] = input("Your answer: ")
    
    return profile

# Main application
if __name__ == "__main__":
    print("\nWelcome to the Digital Nomad Tax Planning Assistant")
    print("This system will analyze your situation and provide tax guidance based on your profile.")
    
    # Collect user information
    user_profile = collect_user_responses()
    
    print("\nGenerating your personalized tax analysis. This may take a moment...")
    
    # Process the profile
    response = graph.invoke({"profile": user_profile})
    
    print("\n" + "="*50)
    print("Your Personalized Tax Analysis")
    print("="*50)
    print("\n" + response["analysis"])
    
    # Option to save the report
    save_option = input("\nWould you like to save this analysis to a file? (y/n): ")
    if save_option.lower() == 'y':
        filename = f"tax_analysis_{user_profile['target_country'].split(' ')[0].lower()}.txt"
        with open(filename, "w") as f:
            f.write("DIGITAL NOMAD TAX ANALYSIS\n")
            f.write("="*30 + "\n\n")
            f.write("USER PROFILE:\n")
            for q in questions:
                f.write(f"{q['question']}: {user_profile[q['id']]}\n")
            f.write("\n" + "="*30 + "\n\n")
            f.write(response["analysis"])
        print(f"Analysis saved to {filename}")
    
    print("\nThank you for using the Digital Nomad Tax Planning Assistant!")
