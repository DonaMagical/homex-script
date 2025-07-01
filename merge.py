from collections import defaultdict
from typing import DefaultDict, Optional, Union
from notification import send_push_notification
from sheet import ProviderItem, ProviderWorkbook
from ai import AIMatchType, AIResponse, GeminiClient
from io_util import file_exists, load_merges_from_json, output_merge_table, save_merges_to_json
from type import ExactCodeMatch, HazyLLMMatch, StrongLLMMatch, ItemMerge, ItemReference, ItemOutput, NoMatch, Provider, sorted_providers
from vector import VectorStore

class Merge:
    CHECKPOINT_INTERVAL = 2
    
    def __init__(self, workbooks: dict[Provider, ProviderWorkbook], reference_provider: Provider, ai_client: GeminiClient, vector_store: VectorStore, checkpoint_file: str = None):
        self.__workbooks = workbooks
        self.reference_provider = reference_provider
        self.__ai_client = ai_client
        self.__vector_store = vector_store
        self.__checkpoint_file = checkpoint_file

    def workbook(self, provider: Provider) -> ProviderWorkbook:
        return self.__workbooks[provider]
    
    @property
    def reference_workbook(self) -> ProviderWorkbook:
        return self.workbook(self.reference_provider)
    
    def get_item_by_ref(self, item_ref: ItemReference) -> ProviderItem:
        workbook = self.workbook(item_ref.provider)
        return workbook.get_item_by_ref(item_ref)
    
    def get_item_by_id_or_code(self, provider: Provider, id: int, code: str) -> ProviderItem | None:
        workbook = self.workbook(provider)
        item = workbook.get_item_by_id(id)
        if item is None:
            print(f"ID lookup failed for {provider}: {id}, doing backup search with code: {code}")
            item = workbook.get_item_by_code(code)
            if item is None:
                print(f"Code lookup failed for {provider}: {code}")
        return item

    def merge(self, output_file: str):
        merges = self.__merge_all()
        coalesced_merges = self.__coalesce_merges(merges)
        output_data = self.__to_output_data(coalesced_merges)
        output_merge_table(output_file, output_data)

    def __merge_all(self) -> list[ItemMerge]:
        [prior_merge_lookup, _] = self.__load_prior_merges()
        item_merges: list[ItemMerge] = []
        for provider in sorted_providers:
            if provider == self.reference_provider:
                print(f"Skipping reference provider: {provider}")
                continue
            workbook = self.workbook(provider) 
            print(f"Merging provider: {provider} ({len(workbook.item_refs)} items)") 
            
            reference_filters = self.__get_reference_filters(provider, item_merges)
            
            for idx, item_ref in enumerate(workbook.item_refs):
                if idx % self.CHECKPOINT_INTERVAL == 0:
                    print(f"Checkpoint: Provider ({provider}) index: {idx} of {len(workbook.item_refs) - 1}")
                    self.__save_checkpoint(item_merges)
                if item_ref in prior_merge_lookup:
                    item_merges.append(prior_merge_lookup[item_ref])
                    continue
                item = self.get_item_by_ref(item_ref)
                try:
                    item_merge = self.__merge_item(item, reference_filters)
                    item_merges.append(item_merge)
                except Exception as e:
                    self.__save_checkpoint(item_merges)
                    send_push_notification("Script Failed", str(e))
                    raise e
        send_push_notification("Script Completed", f"Merged {len(item_merges)} items")
        return item_merges
    
    def __get_reference_filters(self, current_provider: Provider, merges: list[ItemMerge]) -> list[Union[Provider, ItemReference]]:
        ref_filters: list[Union[Provider, ItemReference]] = []
        ordered_providers: list[Provider] = [self.reference_provider] + [p for p in sorted_providers if p != self.reference_provider]
        for provider in ordered_providers:
            if provider == current_provider:
                break
            if provider == self.reference_provider:
                ref_filters.append(provider)
                break
            for merge in merges:
                if isinstance(merge, NoMatch) and merge.query.provider == provider:
                    ref_filters.append(merge.query)

        return ref_filters
    
    def __merge_item(self, item: ProviderItem, reference_filters: list[Union[Provider, ItemReference]]) -> ItemMerge:
        # Exact code match
        reference_workbook = self.workbook(self.reference_provider)
        matching_code_item = reference_workbook.get_item_by_code(item.code)
        if matching_code_item is not None:
            return ExactCodeMatch(
                query=item.to_item_ref(),
                match=matching_code_item.to_item_ref()
            )
        
        # LLM-based match
        return self.__match_with_llm(item, reference_filters)
    
    def __match_with_llm(self, item: ProviderItem, reference_filters: list[Union[Provider, ItemReference]]) -> ItemMerge:
        # First filter for relevant reference items
        relevant_items = self.__vector_store.get_relevant_items(item.to_item_ref(), reference_filters)
        full_reference_items = [self.get_item_by_ref(item_ref) for item_ref in relevant_items]

        match_result = self.__ai_client.generate_match_response_advanced(full_reference_items, item)
        merge = self.__evaluate_llm_match(item, match_result)
        if merge is not None:
            return merge
        
        print(f"Initial AI match didn't return valid item, attempting follow-up match")
        match_result = self.__ai_client.generate_followup_match_response(full_reference_items, item, match_result)
        merge = self.__evaluate_llm_match(item, match_result)
        if merge is not None:
            return merge
        
        raise Exception(f"No match found for {item.provider}: {item.id}, {item.code}")

    def __evaluate_llm_match(self, item: ProviderItem, match_result: AIResponse) -> Optional[ItemMerge]:
        if match_result.type == AIMatchType.NoMatch:
            return NoMatch(query=item.to_item_ref())
        
        match_item = self.get_item_by_id_or_code(match_result.item.provider, match_result.item.id, match_result.item.code)
        if match_item is None:
            return None
        
        if match_result.type == AIMatchType.StrongMatch:
            return StrongLLMMatch(query=item.to_item_ref(), match=match_item.to_item_ref())
        else:
            return HazyLLMMatch(query=item.to_item_ref(), match=match_item.to_item_ref())
    
    def __load_prior_merges(self) -> tuple[dict[ItemReference, ItemMerge], dict[ItemReference, ItemMerge]]:
        if self.__checkpoint_file is None or not file_exists(self.__checkpoint_file):
            return {}, {}
        
        prior_merges = load_merges_from_json(self.__checkpoint_file)
        return index_merges(prior_merges)
    
    def __save_checkpoint(self, merges: list[ItemMerge]) -> None:
        save_merges_to_json(merges, self.__checkpoint_file)

    def __coalesce_merges(self, merges: list[ItemMerge]) -> list[tuple[ItemReference, list[ItemMerge]]]:
        result: list[tuple[ItemReference, list[ItemMerge]]] = []
        [query_to_merge, match_to_merges] = index_merges(merges)

        ordered_providers: list[Provider] = [self.reference_provider] + [p for p in sorted_providers if p != self.reference_provider]
        for provider in ordered_providers:
            workbook = self.workbook(provider)
            for item_ref in workbook.item_refs:
                merged_items = match_to_merges[item_ref] if item_ref in match_to_merges else []
                matched_item = query_to_merge.get(item_ref)
                if matched_item is None or isinstance(matched_item, NoMatch):
                    result.append((item_ref, merged_items))
        
        return result
    
    def __to_output_data(self, coalesced_merges: list[tuple[ItemReference, list[ItemMerge]]]) -> list[list[Optional[ItemOutput]]]:
        output_data: list[list[Optional[ItemOutput]]] = []
        ordered_providers: list[Provider] = [self.reference_provider] + [p for p in sorted_providers if p != self.reference_provider]
        
        for (base_ref, merges) in coalesced_merges:
            merges_by_provider = group_merges_by_provider(merges)
            
            # Find the maximum number of merges for any provider
            max_merges_per_provider = 1
            for provider in ordered_providers:
                if provider != base_ref.provider:
                    merges_for_provider = merges_by_provider[provider]
                    max_merges_per_provider = max(max_merges_per_provider, len(merges_for_provider))
            
            # Create rows for each merge combination
            for merge_index in range(max_merges_per_provider):
                row_data: list[Optional[ItemOutput]] = []
                for provider in ordered_providers:
                    if provider == base_ref.provider:
                        item = self.get_item_by_ref(base_ref)
                        row_data.append(item.to_item_output(None))
                        continue
                    
                    merges_for_provider = merges_by_provider[provider]
                    if merge_index < len(merges_for_provider):
                        merge_for_provider = merges_for_provider[merge_index]
                        item = self.get_item_by_ref(merge_for_provider.query)
                        item_output = item.to_item_output(merge_for_provider.type)
                        row_data.append(item_output)
                    else:
                        row_data.append(None)
                output_data.append(row_data)
        
        return output_data
    
def index_merges(merges: list[ItemMerge]) -> tuple[dict[ItemReference, ItemMerge], dict[ItemReference, ItemMerge]]:
    query_to_merge: dict[ItemReference, ItemMerge] = {}
    match_to_merges: DefaultDict[ItemReference, list[ItemMerge]] = defaultdict(list)
    for merge in merges:
        query_to_merge[merge.query] = merge
        if not isinstance(merge, NoMatch):
            match_to_merges[merge.match].append(merge)
    return query_to_merge, match_to_merges

def group_merges_by_provider(merges: list[ItemMerge]) -> dict[Provider, list[ItemMerge]]:
    result: DefaultDict[Provider, list[ItemMerge]] = defaultdict(list)
    for merge in merges:
        result[merge.query.provider].append(merge)
    return result