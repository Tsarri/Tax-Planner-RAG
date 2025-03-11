import os
import getpass
import datetime
import re
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

# Set up Mistral API key automatically
os.environ["MISTRAL_API_KEY"] = "0OYFp5b31qOfBEYVppnkUlDmn4uuLen4"

# Initialize AI components
print("Initializing AI components...")
llm = ChatMistralAI(
    model="mistral-large-latest",
    timeout=120.0  # Increased timeout to 120 seconds
)
embeddings = MistralAIEmbeddings(model="mistral-embed")
vector_store = InMemoryVectorStore(embeddings)

# Path to your tax documents folder
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
Use a direct, conversational tone as if you're a professional speaking to a colleague.
Avoid overly formal language, jargon, and passive voice.
Use contractions and be straightforward with recommendations.
Keep sections focused on a single idea.
Emphasize practical, actionable advice.

Your analysis should be organized in these sections:
RESIDENCE STATUS: Analysis of tax residency status in both home and target countries
TAX OBLIGATIONS: Specific tax rates and filing requirements
AVAILABLE BENEFITS: Special regimes, deductions, and credits
PLANNING STRATEGIES: Actionable recommendations to optimize tax position
POTENTIAL ISSUES: Risks and considerations to be aware of

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

def generate_fallback_analysis(profile):
    """Generate a basic analysis when the API call fails"""
    country = profile["target_country"]
    stay_duration = profile["stay_duration"]
    income_source = profile["income_source"]
    
    analysis = f"""
RESIDENCE STATUS:
Based on your plan to stay in {country} for {stay_duration}, you may be considered a tax resident depending on local rules. Most countries use a 183-day threshold, but some have additional criteria.

TAX OBLIGATIONS:
As a {income_source}, you'll likely need to pay income tax in your country of residence. Double taxation agreements may apply between your home country and {country}.

AVAILABLE BENEFITS:
{country} may offer special tax regimes for digital nomads or new residents. These could include reduced tax rates or exemptions on foreign-source income.

PLANNING STRATEGIES:
Consider timing your move to maximize tax benefits. Maintain proper documentation of your physical presence and income sources.

POTENTIAL ISSUES:
Be aware of permanent establishment risks if you're running a business. Social security obligations can be complex for international workers.
"""
    return analysis

def generate(state: State):
    # Format retrieved documents for the prompt
    docs_content = "\n\n".join([
        f"[Document from {doc.metadata.get('country')}, Category: {doc.metadata.get('category')}]\n{doc.page_content}"
        for doc in state["context"]
    ])
    
    try:
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
    except Exception as e:
        print(f"Error generating analysis: {e}")
        # Provide a fallback response
        return {"analysis": generate_fallback_analysis(state["profile"])}

# Function to create a professionally formatted PDF report
def create_pdf_report(profile, analysis, filename, questions):
    # Create a PDF document
    doc = SimpleDocTemplate(
        filename, 
        pagesize=letter,
        leftMargin=42, 
        rightMargin=42,
        topMargin=36,
        bottomMargin=36
    )
    
    # Create styles
    styles = getSampleStyleSheet()
    
    # Create professional custom styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=14,
        fontName='Helvetica-Bold',
        leading=16,
        spaceAfter=10,
        alignment=TA_CENTER
    )
    
    subtitle_style = ParagraphStyle(
        'SubtitleStyle',
        parent=styles['Heading2'],
        fontSize=12,
        fontName='Helvetica-Bold',
        leading=14,
        spaceBefore=8,
        spaceAfter=4
    )
    
    section_title_style = ParagraphStyle(
        'SectionTitleStyle',
        parent=styles['Heading3'],
        fontSize=10,
        fontName='Helvetica-Bold',
        leading=12,
        spaceBefore=6,
        spaceAfter=2
    )
    
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontSize=9,
        fontName='Helvetica',
        leading=11,
        spaceAfter=3
    )
    
    # Create bullet point style
    bullet_style = ParagraphStyle(
        'BulletStyle',
        parent=body_style,
        leftIndent=20,
        firstLineIndent=-15,
        spaceAfter=5
    )
    
    # List to hold the PDF elements
    elements = []
    
    # Add title
    elements.append(Paragraph("Digital Nomad Tax Analysis Report", title_style))
    
    # Add date in smaller text
    date_str = datetime.datetime.now().strftime("%B %d, %Y")
    date_style = ParagraphStyle(
        'DateStyle',
        parent=styles['Normal'],
        fontSize=8,
        alignment=TA_CENTER
    )
    elements.append(Paragraph(f"Generated: {date_str}", date_style))
    elements.append(Spacer(1, 8))
    
    # Create a table for the profile
    profile_data = [
        ["Profile Parameters", "Value"],
        ["Citizenship", profile["citizenship"]],
        ["Target Country", profile["target_country"]],
        ["Stay Duration", profile["stay_duration"]],
        ["Income Source", profile["income_source"]],
        ["Annual Income", profile["income_range"]],
        ["Business Structure", profile["business_structure"]],
        ["Special Tax Status", profile["special_tax_interest"]],
        ["Home Office Use", profile["home_office"]],
        ["Long-term Plans", profile["long_term_plans"]]
    ]
    
    profile_table = Table(profile_data, colWidths=[140, 310])
    profile_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (1, 0), 4),
        ('TOPPADDING', (0, 0), (1, 0), 4),
        
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 3),
        ('TOPPADDING', (0, 1), (-1, -1), 3),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    
    elements.append(profile_table)
    elements.append(Spacer(1, 12))
    
    # Process the analysis content for a more natural, professional tone
    # Clean up the analysis text - remove markdown symbols and extra whitespace
    cleaned_analysis = analysis.replace("#", "").replace("*", "")
    cleaned_analysis = re.sub(r'\s+', ' ', cleaned_analysis)
    
    # Define section patterns and titles
    section_patterns = [
        (r"(?i)RESIDENCE\s+STATUS[\s:]+(.+?)(?=TAX OBLIGATIONS|$)", "Residence Status"),
        (r"(?i)TAX\s+OBLIGATIONS[\s:]+(.+?)(?=AVAILABLE BENEFITS|$)", "Tax Obligations"),
        (r"(?i)AVAILABLE\s+BENEFITS[\s:]+(.+?)(?=PLANNING STRATEGIES|$)", "Available Benefits"),
        (r"(?i)PLANNING\s+STRATEGIES[\s:]+(.+?)(?=POTENTIAL ISSUES|$)", "Planning Strategies"),
        (r"(?i)POTENTIAL\s+ISSUES[\s:]+(.+?)(?=$)", "Potential Issues")
    ]
    
    # Add analysis with improved language
    elements.append(Paragraph("Tax Analysis Recommendations", subtitle_style))
    
    found_sections = False
    
    # Function to improve text readability and professionalism
    def improve_text(text):
        # Replace overly formal phrases
        text = re.sub(r'it is recommended that', 'we recommend', text, flags=re.I)
        text = re.sub(r'it is advised that', 'we advise', text, flags=re.I)
        text = re.sub(r'it is suggested that', 'we suggest', text, flags=re.I)
        
        # Replace passive voice with active voice where possible
        text = re.sub(r'should be considered', 'consider', text, flags=re.I)
        text = re.sub(r'could be implemented', 'consider implementing', text, flags=re.I)
        text = re.sub(r'must be included', 'include', text, flags=re.I)
        
        # Make language more direct
        text = re.sub(r'in order to', 'to', text, flags=re.I)
        text = re.sub(r'for the purpose of', 'for', text, flags=re.I)
        text = re.sub(r'in the event that', 'if', text, flags=re.I)
        
        # Improve transitions
        text = re.sub(r'Additionally,', 'Also,', text, flags=re.I)
        text = re.sub(r'Furthermore,', 'Moreover,', text, flags=re.I)
        text = re.sub(r'Consequently,', 'As a result,', text, flags=re.I)
        
        # Remove redundant phrases
        text = re.sub(r'and thus,?', 'and', text, flags=re.I)
        text = re.sub(r'as a matter of fact', '', text, flags=re.I)
        
        # Simplify business jargon
        text = re.sub(r'utilize', 'use', text, flags=re.I)
        text = re.sub(r'facilitate', 'help', text, flags=re.I)
        text = re.sub(r'leverage', 'use', text, flags=re.I)
        
        return text
    
    for pattern, section_title in section_patterns:
        match = re.search(pattern, cleaned_analysis, re.DOTALL)
        if match:
            found_sections = True
            content = match.group(1).strip()
            
            elements.append(Paragraph(section_title, section_title_style))
            
            # Process the content into bullet points
            sentences = re.split(r'(?<=[.!?])\s+', content)
            
            # Group related sentences into bullet points
            bullet_points = []
            current_point = []
            topic_indicators = ['tax', 'resident', 'income', 'rate', 'benefit', 'deduction', 'credit', 'strategy', 'recommend', 'consider']
            
            for i, sentence in enumerate(sentences):
                if len(current_point) == 0:
                    current_point.append(sentence)
                elif i > 0 and any(indicator in sentence.lower() for indicator in topic_indicators) and not any(indicator in sentences[i-1].lower() for indicator in topic_indicators):
                    # New topic starting - create a new bullet point
                    bullet_points.append(' '.join(current_point))
                    current_point = [sentence]
                elif len(' '.join(current_point)) + len(sentence) > 150:  # Keep bullet points concise
                    bullet_points.append(' '.join(current_point))
                    current_point = [sentence]
                else:
                    current_point.append(sentence)
            
            if current_point:
                bullet_points.append(' '.join(current_point))
            
            # Add bullet points to the document
            for point in bullet_points:
                improved_point = improve_text(point)
                elements.append(Paragraph(f"• {improved_point}", bullet_style))
    
    # If no sections were found, process the whole text as bullet points
    if not found_sections:
        # Split by paragraphs and convert each to a bullet point
        paragraphs = cleaned_analysis.split('\n\n')
        for para in paragraphs:
            if para.strip():
                improved_para = improve_text(para.strip())
                elements.append(Paragraph(f"• {improved_para}", bullet_style))
    
    # Build the PDF
    doc.build(elements)
    print(f"PDF report saved as {filename}")

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
        filename = f"tax_analysis_{user_profile['target_country'].split(' ')[0].lower()}.pdf"
        create_pdf_report(user_profile, response["analysis"], filename, questions)
    
    print("\nThank you for using the Digital Nomad Tax Planning Assistant!")
