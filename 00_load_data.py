from datasets import load_dataset

# Load the dataset
ds = load_dataset("kurry/sp500_earnings_transcripts")

save_path = "sp500_transcripts_local"

rec = ds[0]

for seg in rec["structured_content"][:3]:
    # 'seg' is a dictionary, so you can access keys like 'speaker' and 'text'
    print(seg["speaker"], ":", seg["text"][:], "…")
    
    
ds_2024 = ds.filter(lambda example: example['year'] == 2024)


first_1000_samples = ds_2024.select(range(1000))

first_1000_samples.save_to_disk(f"{save_path}/records")