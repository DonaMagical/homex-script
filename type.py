from enum import StrEnum, auto
from dataclasses import dataclass
from typing import Union, Dict, Any

class Provider(StrEnum):
    Haller = auto()
    Gem = auto()
    Universe = auto()
    WeltmanPrinceton = auto()

sorted_providers = sorted(Provider, key=lambda p: p.value)

class SheetName(StrEnum):
    Materials = 'Materials'
    Equipment = 'Equipment'

@dataclass(frozen=True)
class ItemReference:
    provider: Provider
    sheet_name: SheetName
    id: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider.value,
            "sheet_name": self.sheet_name.value,
            "id": self.id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ItemReference':
        return cls(
            provider=Provider(data["provider"]),
            sheet_name=SheetName(data["sheet_name"]),
            id=data["id"]
        )
    
class MatchType(StrEnum):
    ExactCode = auto()
    StrongLLM = auto()
    HazyLLM = auto()
    NoMatch = auto()

@dataclass
class ExactCodeMatch:
    query: ItemReference
    match: ItemReference
    type: MatchType = MatchType.ExactCode
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "query": self.query.to_dict(),
            "match": self.match.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExactCodeMatch':
        return cls(
            query=ItemReference.from_dict(data["query"]),
            match=ItemReference.from_dict(data["match"])
        )

@dataclass
class StrongLLMMatch:
    query: ItemReference
    match: ItemReference
    type: MatchType = MatchType.StrongLLM
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "query": self.query.to_dict(),
            "match": self.match.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrongLLMMatch':
        return cls(
            query=ItemReference.from_dict(data["query"]),
            match=ItemReference.from_dict(data["match"])
        )
    

@dataclass
class HazyLLMMatch:
    query: ItemReference
    match: ItemReference
    type: MatchType = MatchType.HazyLLM
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "query": self.query.to_dict(),
            "match": self.match.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HazyLLMMatch':
        return cls(
            query=ItemReference.from_dict(data["query"]),
            match=ItemReference.from_dict(data["match"])
        )

@dataclass
class NoMatch:
    query: ItemReference
    type: MatchType = MatchType.NoMatch
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "query": self.query.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NoMatch':
        return cls(
            query=ItemReference.from_dict(data["query"])
        )

ItemMerge = Union[ExactCodeMatch, StrongLLMMatch, NoMatch]

@dataclass(frozen=True)
class ItemOutput:
    match_type: MatchType | None
    category_id: str
    category_name: str
    is_inventory: bool
    id: int
    code: str
    name: str
    description: str
    intacct_gl_group: str
    unit_of_measure: str
    cost: float