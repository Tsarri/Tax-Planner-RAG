# Tax Planner for Digital Nomads

This application helps digital nomads understand tax requirements in different countries using Retrieval Augmented Generation (RAG) technology.

## Included Documentation

This repository contains tax documentation for reference purposes. The documents are organized by country and provide information about:
- Digital nomad visa requirements
- Income tax regulations
- Business structure considerations
- Special tax statuses

## How to Use

1. Clone this repository:

git clone https://github.com/yourusername/tax-planner-rag.git

2. Install required dependencies:

pip install -r requirements.txt

3. Set your Mistral AI API key:

export MISTRAL_API_KEY="0OYFp5b31qOfBEYVppnkUlDmn4uuLen4"

4. Run the application:

python tax_planner.py

5. Answer the questions about your situation to receive a personalized tax analysis.

## Setting Up Tax Documentation

This application requires access to tax documentation files. When you clone this repository, the application will look for these files in the `TaxDocuments` folder in the same directory as the application.

The repository includes these documents, organized by country:
- TaxDocuments/Portugal/
- TaxDocuments/Spain/
- TaxDocuments/USA/
- TaxDocuments/[other countries...]

If you want to use your own tax documentation or store it in a different location, you can specify the path when running the application:

python tax_planner.py --docs-path /path/to/your/documents