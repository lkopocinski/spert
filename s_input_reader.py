import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Iterable

from tqdm import tqdm

from spert import util
from spert.entities import Dataset, Document, Relation, Entity, EntityType, RelationType


class SemrelBaseInputReader(ABC):
    def __init__(self, types_path: str):
        types = json.load(open(types_path), object_pairs_hook=OrderedDict)  # entity + relation types

        self._entity_types = OrderedDict()
        self._idx2entity_type = OrderedDict()
        self._relation_types = OrderedDict()
        self._idx2relation_type = OrderedDict()

        # entities
        # add 'None' entity type
        none_entity_type = EntityType('None', 0, 'None', 'No Entity')
        self._entity_types['None'] = none_entity_type
        self._idx2entity_type[0] = none_entity_type

        # specified entity types
        for i, (key, v) in enumerate(types['entities'].items()):
            entity_type = EntityType(key, i + 1, v['short'], v['verbose'])
            self._entity_types[key] = entity_type
            self._idx2entity_type[i + 1] = entity_type

        # relations
        # add 'None' relation type
        none_relation_type = RelationType('None', 0, 'None', 'No Relation')
        self._relation_types['None'] = none_relation_type
        self._idx2relation_type[0] = none_relation_type

        # specified relation types
        for i, (key, v) in enumerate(types['relations'].items()):
            relation_type = RelationType(key, i + 1, v['short'], v['verbose'], v['symmetric'])
            self._relation_types[key] = relation_type
            self._idx2relation_type[i + 1] = relation_type

        self._datasets = {}
        self._context_size = -1

    @abstractmethod
    def read(self, datasets):
        pass

    def get_dataset(self, label) -> Dataset:
        return self._datasets[label]

    def get_entity_type(self, idx) -> EntityType:
        return self._idx2entity_type[idx]

    def get_relation_type(self, idx) -> RelationType:
        return self._idx2relation_type[idx]

    @property
    def datasets(self):
        return self._datasets

    @property
    def entity_types(self):
        return self._entity_types

    @property
    def relation_types(self):
        return self._relation_types

    @property
    def relation_type_count(self):
        return len(self._relation_types)

    @property
    def entity_type_count(self):
        return len(self._entity_types)

    @property
    def context_size(self):
        return self._context_size


class SemrelJsonInputReader(SemrelBaseInputReader):
    def __init__(self, types_path: str):
        super().__init__(types_path)

    def read(self, dataset_paths):
        for dataset_label, dataset_path in dataset_paths.items():
            dataset = Dataset(dataset_label, self)
            self._parse_dataset(dataset_path, dataset)
            self._datasets[dataset_label] = dataset

        self._context_size = self._calc_context_size(self._datasets.values())

    @staticmethod
    def _calc_context_size(datasets: Iterable[Dataset]):
        sizes = []

        for dataset in datasets:
            for doc in dataset.documents:
                sizes.append(len(doc.tokens))

        return max(sizes)

    def _parse_dataset(self, dataset_path, dataset):
        documents = json.load(open(dataset_path))
        for document in tqdm(documents, desc="Parse dataset '%s'" % dataset.label):
            self._parse_document(document, dataset)

    def _parse_document(self, doc, dataset) -> Document:
        jtokens = doc['tokens']
        jrelations = doc['relations']
        jentities = doc['entities']

        # parse tokens
        doc_tokens = self._parse_tokens(jtokens, dataset)

        # parse entity mentions
        entities = self._parse_entities(jentities, doc_tokens, dataset)

        # parse relations
        relations = self._parse_relations(jrelations, entities, dataset)

        return dataset.create_document(doc_tokens, entities, relations, None)

    def _parse_tokens(self, jtokens, dataset):
        doc_tokens = []

        # parse tokens
        for i, token_phrase in enumerate(jtokens):
            span_start = i
            span_end = span_start + 1

            token = dataset.create_token(i, span_start, span_end, token_phrase)
            doc_tokens.append(token)

        return doc_tokens

    def _parse_entities(self, jentities, doc_tokens, dataset) -> List[Entity]:
        entities = []

        for entity_idx, jentity in enumerate(jentities):
            entity_type = self._entity_types[jentity['type']]
            start, end = jentity['start'], jentity['end']

            # create entity mention
            tokens = doc_tokens[start:end]
            phrase = " ".join([t.phrase for t in tokens])
            entity = dataset.create_entity(entity_type, tokens, phrase)
            entities.append(entity)

        return entities

    def _parse_relations(self, jrelations, entities, dataset) -> List[Relation]:
        relations = []

        for jrelation in jrelations:
            relation_type = self._relation_types[jrelation['type']]

            head_idx = jrelation['head']
            tail_idx = jrelation['tail']

            # create relation
            head = entities[head_idx]
            tail = entities[tail_idx]

            reverse = int(tail.tokens[0].index) < int(head.tokens[0].index)

            # for symmetric relations: head occurs before tail in sentence
            if relation_type.symmetric and reverse:
                head, tail = util.swap(head, tail)

            relation = dataset.create_relation(relation_type, head_entity=head, tail_entity=tail, reverse=reverse)
            relations.append(relation)

        return relations


if __name__ == '__main__':
    types_path = "scripts/data/datasets/conll04/conll04_types.json"
    train_path = "scripts/data/datasets/conll04/conll04_train.json"

    reader = SemrelJsonInputReader(types_path)
    reader.read({'train': train_path})

    train_dataset = reader.get_dataset('train')
    print('A')
