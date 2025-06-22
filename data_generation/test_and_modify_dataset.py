from transformers import AutoTokenizer
import json
from pathlib import Path

class TestAndModifyDataset:
    def __init__(self, dataset_path, target_model_name):
        if "train_shareGPT_llama3.2_1B.jsonl" not in dataset_path:
            raise ValueError("This is not a generic class. It is only for this specific dataset.")
        
        self.dataset_path = dataset_path
        self.raw_dataset = self._read_jsonl(Path(dataset_path))
        self.tokenizer = AutoTokenizer.from_pretrained(
            target_model_name,
            padding_side="right",
            use_fast=True,
        )
        
    @staticmethod
    def _read_jsonl(path: Path):
        with path.open() as f:
            return [json.loads(line) for line in f if line.strip()]
        
    def test_tokenization_length(self):
        for i, sample in enumerate(self.raw_dataset):
            prompt = sample["text"]
            tokenized_prompt = self.tokenizer(prompt, return_tensors="pt")
            print(f"Sample {i} length: {tokenized_prompt.input_ids.shape}")
            
            if i > 40:
                break
            
    def get_data_fewshot_prefix(self):
        prefix_hint = "tailor the activities to the birthday child's interests and preferences. Have a great celebration!\n### "
        prefix_hint_len = len(prefix_hint)
        prefix_board = []
        for i, sample in enumerate(self.raw_dataset):
            prompt = sample["text"]
            start_index = prompt.index(prefix_hint)
            end_index = start_index + prefix_hint_len
            
            if prefix_board == []:
                prefix_board.append(len(prompt[:start_index]))
            elif prefix_board[-1] != len(prompt[:start_index]):
                prefix_board.append(len(prompt[:start_index]))
            else:
                pass
            
        print(prefix_board)

    def target_masking(self):
        prefix_hint = "tailor the activities to the birthday child's interests and preferences. Have a great celebration!\n### "
        prefix_hint_len = len(prefix_hint)
        prompts = []
        
        for i, text in enumerate(self.raw_dataset):
            prompt = text["text"]
            prompts.append(prompt)
        
        encoding = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            max_length=4096,
            truncation=True,
            return_offsets_mapping=True,
        )
        
        pass

if __name__ == "__main__":
    dataset_path = "/home/nxclab/wonjun/Medusa/ShareGPT_Vicuna_unfiltered/train_shareGPT_llama3.2_1B.jsonl"
    target_model_name = "meta-llama/Llama-3.2-1B"
    test_and_modify_dataset = TestAndModifyDataset(dataset_path, target_model_name)
    
    # tests
    # test_and_modify_dataset.test_tokenization_length()
    test_and_modify_dataset.get_data_fewshot_prefix()