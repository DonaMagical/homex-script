import argparse
import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from type import Provider, sorted_providers
from merge import Merge
from sheet import load_workbooks
from vector import VectorStore
from ai import GeminiClient

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description='HomeX Merge Script')
    parser.add_argument('--api-key', help='Google Gemini API Key')
    parser.add_argument(
        '--reference-provider',
        default=Provider.Haller.value, 
        choices=[provider.value for provider in sorted_providers],
        help='Reference provider (default: Haller)'
    )
    parser.add_argument(
        '--checkpoint-file',
        default='output/merges.json', 
        help='Input JSON file for loading & incrementally saving merges (default: output/merges.json)'
    )
    parser.add_argument(
        '--output-file',
        default='output/merge-table.xlsx', 
        help='Output Excel file (default: output/merge-table.xlsx)'
    )
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Error: API key must be provided via --api-key argument or GEMINI_API_KEY environment variable")
        sys.exit(1)

    ai_client = GeminiClient(api_key)

    qdrant_url = os.getenv('QDRANT_API_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    if not qdrant_url or not qdrant_api_key:
        print("Error: QDRANT_API_URL and QDRANT_API_KEY must be provided via environment variables")
        sys.exit(1)
    
    qdrant_client = QdrantClient(
        os.getenv('QDRANT_API_URL'),
        api_key=os.getenv('QDRANT_API_KEY')
    )
    vector_store = VectorStore(qdrant_client)

    workbooks = load_workbooks()

    # vector_store.store_embeddings(ai_client, workbooks) # already done

    merger = Merge(workbooks, Provider.Haller, ai_client, vector_store, args.checkpoint_file)
    merger.merge(args.output_file)

    print("Done!")
    

if __name__ == "__main__":
    main()
