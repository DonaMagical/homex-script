from google import genai
from google.genai import types, errors
import json
from pydantic import BaseModel, model_validator
from typing import List, Optional
import yaml
from sheet import ProviderItem
from type import Provider
from enum import StrEnum, auto
import time

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
    
    @model_validator(mode='after')
    def validate_match_consistency(self) -> 'AIResponse':
        if self.type in [AIMatchType.StrongMatch, AIMatchType.HazyMatch] and self.item is None:
            raise ValueError(f"When match type is {self.type}, item must be provided")
        if self.type == AIMatchType.NoMatch and self.item is not None:
            raise ValueError("When match type is NoMatch, item should be None")
        return self

def items_to_yaml(items: List[ProviderItem]) -> str:
    yaml_data = []
    
    for item in items:
        item_dict = {
            'Provider': item.provider.value,
            'Id': item.id,
            'Code': item.code,
            'Name': item.get('Name'),
            'Description': item.get('Description')
        }
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
- Product code(s) may appear in any of the data fields that can help identify/differentiate items
- Since there may be multiple codes listed, only perform code comparisons if they have similar schemas/formats
- When product code(s) don't suggest a match, analyze the item type, function, brand, and/or specs to see if it's still a match
- Assume that similar parts with different dimensions (even if slight) correspond to different items
- Generic items (i.e. containing non-descriptive text referring to a class of items, e.g. "Microwaves"): Query items like this should return no match, and other query items should not be matched to such generic items in the master list
- Don't match with generic entries intended to be a catch-all (e.g. just "Microwaves")
- Use the provided terminology key to help normalize descriptions

Expected Output:
Look for the best matching item in the master list, and return one of the the following:
```
{{ 
    // Match found with high confidence (product codes with exact full/subset match and other mutual specs all match)
    "type": "{AIMatchType.StrongMatch.value}",
    "item": {{ ... }} // Best matching entry from master list (CRITICAL: ensure provider name, ID, and code match exactly)
}}

{{ 
    // Possible match found with less confidence (be careful to ensure that the ID/code match the item in the master list)
    "type": "{AIMatchType.HazyMatch.value}",
    "item": {{ ... }} // Best matching entry from master list (CRITICAL: ensure provider name, ID, and code match exactly)}}

{{ 
    // No match found (shared product codes with similar schemas don't match and/or other mutual specs don't match)
    "type": "{AIMatchType.NoMatch.value}",
    "item": null
}}
```

IMPORTANT: Only return items that actually exist in the master list provided
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
    def __init__(self, api_key: str):
        self.__client = genai.Client(api_key=api_key)

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
                            thinking_budget=-1,
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
