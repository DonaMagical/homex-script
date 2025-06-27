import json
from itertools import chain
import os
from typing import List, Dict, Any, Optional
from type import ItemMerge, ItemOutput, MatchType, ExactCodeMatch, HazyLLMMatch, Provider, StrongLLMMatch, NoMatch
from openpyxl import Workbook

def file_exists(filename: str) -> bool:
    return os.path.exists(filename)

def deserialize_merge(data: Dict[str, Any]) -> ItemMerge:
    merge_type = data.get("type")
    
    if merge_type == MatchType.ExactCode:
        return ExactCodeMatch.from_dict(data)
    elif merge_type == MatchType.StrongLLM:
        return StrongLLMMatch.from_dict(data)
    elif merge_type == MatchType.HazyLLM:
        return HazyLLMMatch.from_dict(data)
    elif merge_type == MatchType.NoMatch:
        return NoMatch.from_dict(data)
    else:
        raise ValueError(f"Unknown merge type: {merge_type}")

def save_merges_to_json(merges: List[ItemMerge], filename: str) -> None:
    with open(filename, 'w') as f:
        json.dump([merge.to_dict() for merge in merges], f, indent=2)

def load_merges_from_json(filename: str) -> List[ItemMerge]:
    with open(filename, 'r') as f:
        data = json.load(f)
    return [deserialize_merge(item) for item in data]

def to_merge_table_match_type(match_type: Optional[MatchType]) -> str:
    match match_type:
        case None:
            return "(Original)"
        case MatchType.ExactCode | MatchType.StrongLLM:
            return "Matched"
        case MatchType.HazyLLM:
            return "Hazy Match"

def to_merge_table_row_section(section_input: Optional[ItemOutput]) -> list:
    if section_input is None:
        return ['No Match'] + [None] * 10
    
    return [
        to_merge_table_match_type(section_input.match_type),
        section_input.category_id,
        section_input.category_name,
        'INV' if section_input.is_inventory else 'NI',
        section_input.id,
        section_input.code,
        section_input.name,
        section_input.description,
        section_input.intacct_gl_group,
        section_input.unit_of_measure,
        section_input.cost,
    ]

def to_merge_table_row(row_input: list[Optional[ItemOutput]]) -> list:
    return list(chain.from_iterable(
        to_merge_table_row_section(item_data) for item_data in row_input
    ))

def output_merge_table(file_name: str, output_data: list[list[Optional[ItemOutput]]]):
    print(f"Outputting merge table to {file_name}")

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Merge Result"
    
    section_headers = [
        "Match Type",
        "Category ID",
        "Category Name",
        "Inventory/Non-Inventory?",
        "ID", 
        "Code",
        "Name",
        "Description",
        "Intacct GL Group",
        "UOM",
        "Cost"
    ]
    sheet.append(section_headers * len(output_data[0]))
    
    for index, row_data in enumerate(output_data):
        row = to_merge_table_row(row_data)
        sheet.append(row)
        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1} of {len(output_data)} rows")
    
    workbook.save(file_name)
    workbook.close()