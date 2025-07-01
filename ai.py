from dataclasses import dataclass
from enum import StrEnum, auto
from google import genai
from google.genai import types, errors
import json
from pydantic import BaseModel, model_validator
import time
from typing import List, Optional
import yaml
from io_util import add_to_revisit_list
from notification import send_push_notification
from sheet import ProviderItem
from type import Provider, SheetName

class MatchingItem(BaseModel):
    provider: Provider  
    id: int
    code: str
    name: str

class AIMatchType(StrEnum):
    StrongMatch = auto()
    HazyMatch = auto()
    NoMatch = auto()

class AIResponse(BaseModel):
    type: AIMatchType
    item: Optional[MatchingItem] = None
    reasoning: str
    
    @model_validator(mode='after')
    def validate_match_consistency(self) -> 'AIResponse':
        if self.type in [AIMatchType.StrongMatch, AIMatchType.HazyMatch] and self.item is None:
            raise ValueError(f"When match type is {self.type}, item must be provided")
        if self.type == AIMatchType.NoMatch and self.item is not None:
            raise ValueError("When match type is NoMatch, item should be None")
        return self

MAX_EMBED_CHUNK_SIZE = 100  # Max embeddings per request

@dataclass
class ItemEmbedResult:
    item: ProviderItem
    embedding: list[float]

def get_embedding_content(item: ProviderItem) -> str:
    if item.sheet_name == SheetName.Equipment:
        return f"""
Category: {item.get('Category.Name') or 'N/A'}
Name: {item.get('Name')}
Description: {item.get('Description')}
Code: {item.code}
Brand: {item.get('Brand') or 'N/A'}
Manufacturer: {item.get('Manufacturer') or 'N/A'}
Model: {item.get('Model') or 'N/A'}
"""
    
    return f"""
Category: {item.get('Category.Name') or 'N/A'}
Name: {item.get('Name')}
Description: {item.get('Description')}
Code: {item.code}
"""

def items_to_yaml(items: List[ProviderItem]) -> str:
    yaml_data = []
    
    for item in items:
        item_dict = {
            'Provider': item.provider.value,
            'Id': item.id,
            'Code': item.code,
        }
        
        if value := item.get('Category.Name'):
            item_dict['Category'] = value
        if value := item.get('Name'):
            item_dict['Name'] = value
        if value := item.get('Description'):
            item_dict['Description'] = value
        
        if item.sheet_name == SheetName.Equipment:
            if value := item.get('Type'):
                item_dict['Type'] = value
            if value := item.get('Brand'):
                item_dict['Brand'] = value
            if value := item.get('Manufacturer'):
                item_dict['Manufacturer'] = value
            if value := item.get('Model'):
                item_dict['Model'] = value
        
        yaml_data.append(item_dict)
    
    return yaml.dump(yaml_data, default_flow_style=False, sort_keys=False)


def item_to_json(item: ProviderItem) -> str:
    return json.dumps({
        'Id': item.id,
        'Code': item.code,
        'Name': item.get('Name'),
        'Description': item.get('Description')
    }, indent=2)


def generate_prompt_messages(reference_items: List[ProviderItem], query_item: ProviderItem) -> List[types.Content]:
    reference_yaml = items_to_yaml(reference_items)
    query_json = item_to_json(query_item)
    
    return [
        types.Content(
            role="user",
            parts=[
                types.Part(text=f"""
Your goal is to consolidate inventory items into an ongoing master list.
You will be prompted with a query item, for which you must determine whether it corresponds to an existing entry in the master list, or is a new item that should be added to the master list.

You will be provided with the following resources:
- TERMINOLOGY KEY: YAML file containing standard terms and their alternative forms
- MASTER INVENTORY LIST: YAML file containing all master inventory items
- QUERY ITEM: JSON object representing the item to be matched

Matching Guidelines:
- ID fields for items are different between providers, so they should not be used for comparison
- Product code(s) may appear in any of the other data fields that can help identify/differentiate items
- Since there may be multiple codes listed, only perform code comparisons if they have similar schemas/formats
- When product code(s) don't suggest a match, analyze the item type, function, brand, and/or specs to see if it's still a match
- Assume that similar parts with different dimensions (even if slight) correspond to different items
- Generic items (i.e. containing non-descriptive text referring to a class of items, e.g. "Microwaves"): Query items like this should return no match, and other query items should not be matched to such generic items in the master list
- Don't match with generic entries intended to be a catch-all (e.g. just "Microwaves")
- Use the provided terminology key to help normalize descriptions
- CRITICAL: If you deterimine there is no match, return a NoMatch response (format shown below)

Expected Output:
Look for the best matching item in the master list, and you must return a JSON object corresponding to one of the three following formats:
```
{{ 
    // Match found with high confidence (product codes with exact full/subset match and other mutual specs all match)
    "type": "{AIMatchType.StrongMatch.value}",
    "item": {{ ... }} // Best matching entry from master list (CRITICAL: ensure provider name, ID, and code match exactly)
    "reasoning": "..." // Brief explanation of why this is a strong match for the query item
}}

{{ 
    // Possible match found with less confidence (no product codes with exact full/subset match, but other mutual specs generally match)
    "type": "{AIMatchType.HazyMatch.value}",
    "item": {{ ... }} // Best matching entry from master list (CRITICAL: ensure provider name, ID, and code match exactly)}}
    "reasoning": "..." // Brief explanation of why this is a hazy match for the query item
{{ 
    // No match found (shared product codes with similar schemas don't match and/or other mutual specs don't match)
    "type": "{AIMatchType.NoMatch.value}",
    "item": null,
    "reasoning": "..." // Brief explanation of why the query item has no match
}}
```

IMPORTANT: If you return a match, only return an item that actually exist in the provided master list
""")
            ]
        ),

        types.Content(
            role="user",
            parts=[
                types.Part(
                    text="=== TERMINOLOGY KEY (YAML) ==="
                ),
                types.Part(
                    inline_data=types.Blob(
                        mime_type="text/plain",
                        data=open("data/terminology.yaml", "rb").read()
                    )
                )
            ]
        ),

        types.Content(
            role="user",
            parts=[
                types.Part(
                    text="=== MASTER INVENTORY LIST (YAML) ==="
                ),
                types.Part(
                    text=reference_yaml
                )
            ]
        ),

        types.Content(
            role="user",
            parts=[
                types.Part(
                    text="=== QUERY ITEM (JSON) ==="
                ),
                types.Part(
                    text=query_json
                )
            ]
        )
    ]


class GeminiClient:
    CHUNK_SIZE = 4_000

    def __init__(self, api_key: str):
        self.__client = genai.Client(api_key=api_key)


    def embed_chunk(self, items: list[ProviderItem], retry_count: int = 0) -> list[ItemEmbedResult]:
        MAX_RETRIES = 2

        if len(items) > MAX_EMBED_CHUNK_SIZE:
            raise Exception(f"Chunk size {len(items)} is greater than max chunk size {MAX_EMBED_CHUNK_SIZE}")
        
        try:
            embedding_contents = [get_embedding_content(item) for item in items]
            embed_response = self.__client.models.embed_content(
                model="text-embedding-004",
                contents=embedding_contents,
                config=types.EmbedContentConfig(task_type="semantic_similarity")
            )
            
            if embed_response.embeddings is None:
                raise Exception(f"No embeddings returned for chunk")
            
            results = []
            for item, embedding in zip(items, embed_response.embeddings):
                if embedding.values is None:
                    raise Exception(f"No embedding values returned for item ({item.provider} {item.id})")
                results.append(ItemEmbedResult(item=item, embedding=embedding.values))
            
            return results
            
        except errors.ClientError as e:
            if e.code == 429 and retry_count < MAX_RETRIES:
                sleep_time = 60 # Gemini request quota is per minute
                print(f"Rate limit exceeded for chunk, sleeping for {sleep_time}s before retry {retry_count + 1}/{MAX_RETRIES}...")
                time.sleep(sleep_time)
                return self.embed_chunk(items, retry_count + 1)
            else:
                raise e
        except Exception as e:
            raise e

    def generate_match_response_chunked(self, reference_items: List[ProviderItem], query_item: ProviderItem) -> AIResponse:
        if len(reference_items) <= self.CHUNK_SIZE:
            return [self.generate_match_response(reference_items, query_item)]
        
        reference_item_chunks = [reference_items[i:i + self.CHUNK_SIZE] for i in range(0, len(reference_items), self.CHUNK_SIZE)]
        chunk_responses = [self.generate_match_response(chunk, query_item) for chunk in reference_item_chunks]
        matching_chunk_responses = [response for response in chunk_responses if response.type != AIMatchType.NoMatch]

        match len(matching_chunk_responses):
            case 0:
                return chunk_responses[0] # Unanimous NoMatch
            case 1:
                return matching_chunk_responses[0] # Single match
            case _:
                candidate_items = []
                for response in matching_chunk_responses:
                    item_found = False
                    for item in reference_items:
                        if item.provider == response.item.provider and (item.id == response.item.id or item.code == response.item.code):
                            candidate_items.append(item)
                            item_found = True
                    if not item_found:
                        print(f"WARNING: Item {response.item.provider}: {response.item.id}, {response.item.code} not found in reference items")
                if len(candidate_items) == 0:
                    raise Exception(f"No matching items found in reference items")    
                return self.generate_match_response(candidate_items, query_item)

    def generate_match_response(self, reference_items: List[ProviderItem], query_item: ProviderItem) -> AIResponse:
        response = self.__client.models.generate_content(
                model="gemini-2.0-flash",
                contents=generate_prompt_messages(reference_items, query_item),
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    response_schema=AIResponse
                )
            )
        return AIResponse.model_validate_json(response.text)
    
    def generate_match_response_advanced(self, reference_items: List[ProviderItem], query_item: ProviderItem) -> AIResponse:
        thinking_budget = 18000
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                response = self.__client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=generate_prompt_messages(reference_items, query_item),
                    config=types.GenerateContentConfig(
                        temperature=0,
                        response_mime_type="application/json",
                        response_schema=AIResponse,
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=thinking_budget if attempt < max_retries else 0,
                            include_thoughts=False
                        )
                    )
                )
                print(f"response.text: {response.text}")          
                
                # Sometimes the API randomly returns None, so retry
                if response.text is None:
                    send_push_notification("Script Warning", f"API returned None response for item {query_item.provider}:{query_item.id}")
                    print(f"response: {response}")
                    if attempt < max_retries:
                        thinking_budget -= 3000
                        print(f"API returned None response for item {query_item.provider}:{query_item.id}, retrying...")
                        continue
                    else:
                        raise Exception(f"API returned None response after {max_retries + 1} attempts for item {query_item.provider}:{query_item.id}")
                
                return AIResponse.model_validate_json(response.text)

            except errors.ClientError as e:
                if e.code == 429 and attempt < max_retries:
                    retry_delay = 30 # Default fallback
                    print(f"Rate limited (429). Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(retry_delay)
                    continue
                else:
                    raise e
            except Exception as e:
                raise e
            
        add_to_revisit_list(query_item.to_item_ref())
        return AIResponse(type=AIMatchType.NoMatch, reasoning="Fallback NoMatch")

    def generate_followup_match_response(self, reference_items: List[ProviderItem], query_item: ProviderItem, previous_response: AIResponse) -> AIResponse:
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                response = self.__client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        *generate_prompt_messages(reference_items, query_item),
                        types.Content(
                            role="model",
                            parts=[
                                types.Part(text="=== Previous Response ==="),
                                types.Part(text=previous_response.model_dump_json(indent=2))
                            ]
                        ),
                        types.Content(
                            role="user",
                            parts=[
                                types.Part(text=f"""
CRITICAL: The previous match suggestion failed because the suggested item (Provider: {previous_response.item.provider}, ID: {previous_response.item.id}, Code: {previous_response.item.code}) does not exist in the master list.

This means either:
1. The provider/ID and provider/code combinations are incorrect
2. The item was suggested but doesn't actually exist in the reference data

Please re-evaluate the query item and find another match from the master list. CRITICAL criteria:
- Check that the provider/ID/code combination all exist for that entry in the master list
- Return a different match from the master list (or no match otherwise)

The query item you're trying to match is: {item_to_json(query_item)}
""")
                            ]
                        )
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0,
                        response_mime_type="application/json",
                        response_schema=AIResponse,
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=15000,
                            include_thoughts=False
                        )
                    )
                )
                return AIResponse.model_validate_json(response.text)
                
            except errors.ClientError as e:
                if e.code == 429 and attempt < max_retries:
                    # TO FIX: Parse retry delay from error
                    retry_delay = 30 # Default fallback
                    print(f"e.details: {e.details}")
                    try:
                        if 'details' in e.details:
                            for detail in e.details['details']:
                                if detail.get('@type') == 'type.googleapis.com/google.rpc.RetryInfo':
                                    retry_delay_str = detail.get('retryDelay', '30s')
                                    retry_delay = int(retry_delay_str.rstrip('s'))
                                    break
                    except (KeyError, ValueError, TypeError):
                        retry_delay = 30 
                    
                    print(f"Rate limited (429). Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(retry_delay)
                    continue
                else:
                    # Re-raise non-429 errors or if max retries exceeded
                    raise
