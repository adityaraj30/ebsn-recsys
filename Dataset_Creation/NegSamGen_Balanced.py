import pandas as pd
import numpy as np

def data_init():
    meta_events = pd.read_csv('*/meta-events.csv')
    meta_members = pd.read_csv('*/meta-members.csv')
    rsvps = pd.read_csv('*/rsvps.csv')
    return meta_events, meta_members, rsvps

def data_gen(rsvps, meta_events, meta_members):
    unique_event_ids = meta_events['event_id'].unique()
    unique_member_ids = meta_members['member_id'].unique()
    existing_pairs = set(zip(rsvps['member_id'], rsvps['event_id']))
    
    new_pairs = []
    
    # Count the number of label 1 entries for each member
    label_1_counts = rsvps[rsvps['label'] == 1]['member_id'].value_counts()

    # Generate negative samples for each member equal to the count of label 1's
    for member_id in unique_member_ids:
        label_1_count = label_1_counts.get(member_id, 0)
        if label_1_count > 0:
            count = 0
            while count < label_1_count:
                event_id = np.random.choice(unique_event_ids)
                if (member_id, event_id) not in existing_pairs and (member_id, event_id) not in new_pairs:
                    new_pairs.append((member_id, event_id))
                    count += 1
    
    # Convert the new pairs to a DataFrame
    new_data = pd.DataFrame(new_pairs, columns=['member_id', 'event_id'])
    new_data['label'] = 0
    
    # Add an 'id' column to the new data
    last_id = rsvps['id'].max()
    new_data['id'] = range(last_id + 1, last_id + 1 + len(new_data))
    
    return new_data

def main():
    meta_events, meta_members, rsvps = data_init()
    data_ret = data_gen(rsvps, meta_events, meta_members)
    combined_data = pd.concat([rsvps, data_ret], ignore_index=True)
    combined_data = combined_data.sample(frac=1).reset_index(drop=True)
    combined_data.to_csv('combined_data_membal.csv', index=False)

if __name__ == "__main__":
    main()
