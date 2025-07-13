# File created by group 6 to avoid cluttering bluehex_main with larger code segments
import pandas as pd

def sample_helper(real_df, synth_df):

    rs = 42
    df_arr = []
    # Find the families value counts
    real_counts = real_df["y_label"].value_counts()
    
    max_family_size = int(real_counts.max() * 1.5)
    
    # ARG1
    real_item_counts = real_counts.items()

    # Balance families as much as possible
    for key, count in real_item_counts:
        
        # Find how many synth samples from that family are needed
        synth_needed = max_family_size - count # ARG2
        
        synth_sub_df = synth_df[synth_df["y_label"] == key]

        df_arr.append(synth_sub_df.sample(n=min(synth_needed,len(synth_sub_df)), random_state=rs))

    filtered = pd.concat(df_arr, ignore_index=True)
    return filtered

# V1: real_counts.max() max_family_size-count
# V2: 600 int((max_family_size - count) / 2)
# V3: real_counts.max() int((max_family_size - count) / 2)
# V4: real_counts.max()*2 max_family_size-count
# V5: int(real_counts.max()*1.5) max_family_size-count