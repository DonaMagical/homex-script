import argparse
import os
import sys
from dotenv import load_dotenv
from ai import GeminiClient
from type import Provider, sorted_providers
from merge import Merge

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

    gemini_client = GeminiClient(api_key)

    reference_provider = Provider(args.reference_provider)

    merger = Merge(reference_provider, gemini_client, args.checkpoint_file)
    merger.merge(args.output_file)

    print("Done!")
    

if __name__ == "__main__":
    main()
