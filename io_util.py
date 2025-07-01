import json
from itertools import chain
import os
from typing import List, Dict, Any, Optional
from type import ItemMerge, ItemOutput, ItemReference, MatchType, ExactCodeMatch, HazyLLMMatch, Provider, StrongLLMMatch, NoMatch
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

def add_to_revisit_list(item_ref: ItemReference, filename: str = "output/items-to-revisit.txt") -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(f"{item_ref.provider},{item_ref.sheet_name},{item_ref.id}\n")

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
        str(section_input.category_id) if section_input.category_id is not None else '',
        str(section_input.category_name) if section_input.category_name is not None else '',
        'INV' if section_input.is_inventory else 'NI',
        section_input.id,
        str(section_input.code) if section_input.code is not None else '',
        str(section_input.name) if section_input.name is not None else '',
        str(section_input.description) if section_input.description is not None else '',
        str(section_input.intacct_gl_group) if section_input.intacct_gl_group is not None else '',
        str(section_input.unit_of_measure) if section_input.unit_of_measure is not None else '',
        float(section_input.cost) if section_input.cost is not None else 0.0,
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