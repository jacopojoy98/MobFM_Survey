import json
import re
import numpy as np


# Load the index to ID mapping from a JSON file. 
# The original filename suggests it's related to trajectory data with 1 million entries from 4 cities.
index2id = json.load(open("/traj_data/location_best_loss_1M_4_city.index.json",'r'))

index2id_final = {}
for key,value in index2id.items():
    final_value = ''
    for i in  value:
        final_value += i
    index2id_final[final_value] = int(key)

print(len(index2id_final))

def get_answer(input_string):
    # Use regular expressions to extract the content between <answer> and </answer>
    match = re.search(r'<answer>(.*?)</answer>', input_string, re.DOTALL)

    if match:
        answer_content = match.group(1)  # Extract the content between <answer> and </answer>
        # print(answer_content)
        return answer_content
    else:
        # print("Content between <answer> tags not found")
        return "00"


time2id  = {"00:00": 0,"00:30": 1,"01:00": 2,"01:30": 3,"02:00": 4,"02:30": 5,"03:00": 6,"03:30": 7,"04:00": 8,"04:30": 9,"05:00": 10,"05:30": 11,"06:00": 12,"06:30": 13,"07:00": 14,"07:30": 15,"08:00": 16,"08:30": 17,"09:00": 18,"09:30": 19,"10:00": 20,"10:30": 21,"11:00": 22,"11:30": 23,"12:00": 24,"12:30": 25,"13:00": 26,"13:30": 27,"14:00": 28,"14:30": 29,"15:00": 30,"15:30": 31,"16:00": 32,"16:30": 33,"17:00": 34,"17:30": 35,"18:00": 36,"18:30": 37,"19:00": 38,"19:30": 39,"20:00": 40,"20:30": 41,"21:00": 42,"21:30": 43,"22:00": 44,"22:30": 45,"23:00": 46,"23:30": 47 }


def extract_time_and_locations(text):
    # Match format: at 11:30 visited location <a_250><b_199><c_242><d_166>
    pattern = r"at (\d{2}:\d{2}) visited location ((<a_\d+><b_\d+><c_\d+><d_\d+>))"
    matches = re.findall(pattern, text)
    times = [match[0] for match in matches]
    locations = [match[1] for match in matches]
    return times, locations

# Store the final results
final_results = []

path = 'generate_1k'




count = 0
"Alignment Model-SFT"
# count_list = np.load('/code/RL_behavior/data_process/get_better_new_token_emb/count_test_api_verson1_final_100_sft.npy')
# # Read the .jsonl file (one JSON object per line)
# with open(f"/code/LLaVA-main_traj/eval_result_0425/{path}/eval_tot.jsonl", "r", encoding="utf-8") as f:
#     for line in f:



#         count += 1
#         if count-1 not in count_list:
#             continue

#         # if count >1200 or count<=1000 :
#         #     continue

#         entry = json.loads(line.strip())  # Parse JSON line by line
#         time_label, loc_label = extract_time_and_locations(entry["label"])
#         time_pred, loc_pred = extract_time_and_locations(entry["predict"])

#         result = {
#             "prompt": int(entry["prompt"]),
#             "time_label": [time2id[key] if key in time2id else -1 for key in time_label],
#             "loc_label": [index2id_final[key] if key in index2id_final else -1 for key in loc_label],
#             "time_pred": [time2id[key] if key in time2id else -1 for key in time_pred],
#             "loc_pred": [index2id_final[key] if key in index2id_final else -1 for key in loc_pred]
#             }

#         final_results.append(result)



def extract_time_and_location_rl(text):
    # Handle two formats: 1) Time and location starting with 'At' or 'at', 2) Direct time and location
    pattern = r'''
        (?:-?\s*(\d{2}:\d{2})\s*[^\w<>]*<a_\d+><b_\d+><c_\d+><d_\d+>)  # List item format
        |                                                              # or
        (?:[Aa]t\s+(\d{2}:\d{2})\s+visited\s+location\s+<a_\d+><b_\d+><c_\d+><d_\d+>)  # At/at format
        |                                                              # or
        (?:\s*(\d{2}:\d{2})\s+visited\s+location\s+<a_\d+><b_\d+><c_\d+><d_\d+>)  # Handle format without "At"
    '''
    
    # Find all matches, use VERBOSE for readability and IGNORECASE to ignore case
    matches = re.findall(pattern, text, re.VERBOSE | re.IGNORECASE)
    
    times = []
    locations = []
    
    # Process the match results
    for match in matches:
        # Extract time (the three formats are in different capturing groups)
        time = match[0] if match[0] else match[1] if match[1] else match[2]
        
        if time:
            # Extract the corresponding location
            loc_pattern = r'<a_\d+><b_\d+><c_\d+><d_\d+>'
            # Find the location in the context that contains the current time
            context = re.search(rf'{re.escape(time)}.*?{loc_pattern}', text, re.IGNORECASE)
            if context:
                loc_match = re.search(loc_pattern, context.group())
                if loc_match:
                    times.append(time)
                    locations.append(loc_match.group())
    
    return times, locations


def extract_time_and_location_rl(text):
    text = re.sub(r'\(.*?\)', '', text)

    # Split the text into paragraphs using newline characters
    paragraphs = text.split('\n')

    # Store the extracted times and locations
    extracted_times = []
    extracted_locations = []

    # Regular expression to match time (HH:MM format) and location (<a_num><b_num><c_num><d_num> format)
    time_pattern = r'\b\d{2}:\d{2}\b'  # Match time in HH:MM format
    location_pattern = r'<a_\d+><b_\d+><c_\d+><d_\d+>'  # Match location ID

    # Iterate through each paragraph to extract time and location
    for paragraph in paragraphs:
        # Remove extra whitespace
        paragraph = paragraph.strip()

        # If the paragraph contains both time and location
        time_matches = re.findall(time_pattern, paragraph)
        location_matches = re.findall(location_pattern, paragraph)
        
        # Ensure the paragraph contains both time and location
        if time_matches and location_matches:
            for time, location in zip(time_matches, location_matches):
                # Exclude times with the "At" prefix
                if not time.startswith("At"):
                    extracted_times.append(time)
                    extracted_locations.append(location)

    # Ensure the number of times and locations are consistent
    if len(extracted_times) != len(extracted_locations):
        raise ValueError("The number of times and locations does not match!")

    return extracted_times, extracted_locations


"Results generated after GRPO"

import numpy as np
count_list = []

print(len(count_list))
print(count_list)
# print(len(data))

count = 0
with open(f"qwen3_4b_traj_rl_predict_no_think/generated_predictions.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        # if count not in count_list:
        #     count += 1
        #     continue
        entry = json.loads(line.strip())  # Parse JSON line by line
        # entry = line

        if  get_answer(entry["predict"]) == "00":
            continue
        # print('123')
        # count_list.append(int(entry['id']))
        # print(entry.keys())
        try:
            time_label, loc_label = extract_time_and_location_rl(get_answer(entry['label']).split('Modified User Trajectory')[1])
        except:
            time_label, loc_label = extract_time_and_location_rl(get_answer(entry['label']).split('Final User Trajectory')[1])

        try:
  
            time_pred, loc_pred = extract_time_and_location_rl(get_answer(entry["predict"]).split('Modified User Trajectory')[1])
        except:

            time_pred, loc_pred = extract_time_and_location_rl(get_answer(entry["predict"]).split('Final User Trajectory')[1])


        if  len(time_pred) != len(loc_pred):
            print('error',len(time_pred) ,len(loc_pred))
            # print()
        if len(time_pred) != len(time_label):
            print('error-- pre_label',len(time_pred) ,len(time_label))

        if len(time_pred) == 0:
            print('time-error')
            continue

        result = {
            # "prompt": int(entry["prompt"]),
            "time_label": [time2id[key] if key in time2id else -1 for key in time_label],
            "loc_label": [index2id_final[key] if key in index2id_final else -1 for key in loc_label],
            "time_pred": [time2id[key] if key in time2id else -1 for key in time_pred],
            "loc_pred": [index2id_final[key] if key in index2id_final else -1 for key in loc_pred]
            }

        final_results.append(result)
        count += 1
print(len(count_list))



print(len(final_results),'final_results')
print(final_results[0])






import numpy as np
import pandas as pd
import json
import os
import warnings
from tqdm import tqdm
from scipy.stats import ks_2samp, wasserstein_distance, anderson
import nltk
from nltk.translate.bleu_score import sentence_bleu

def kl_divergence(p, q):   #KL
    """Calculate KL divergence"""
    p = np.clip(p, 1e-10, None)
    q = np.clip(q, 1e-10, None)
    return np.sum(p * np.log(p / q))

def evaluate_similarity(real_data, synthetic_data):    #AD
    """Calculate the distribution difference between real and synthetic data, and verify similarity through the AD test"""
    # Calculate frequency distribution
    real_counts = pd.Series(real_data).value_counts(normalize=True)
    synthetic_counts = pd.Series(synthetic_data).value_counts(normalize=True)

    # Align the two frequency distributions
    all_intents = real_counts.index.union(synthetic_counts.index)
    real_probs = real_counts.reindex(all_intents, fill_value=0)
    synthetic_probs = synthetic_counts.reindex(all_intents, fill_value=0)

    # Calculate frequency difference
    diff = real_probs - synthetic_probs

    # Perform Anderson-Darling test
    ad_stat = anderson(diff, dist='norm').statistic
    return ad_stat

def bleu_score(real_data, synthetic_data):   #BLEU
    """Calculate BLEU score"""
    bleu = sentence_bleu([real_data], synthetic_data, smoothing_function = nltk.translate.bleu_score.SmoothingFunction().method1)
    return bleu
def distinct_n_score(sequence, n = 2):
    """
    Calculate the Distinct-n metric
    :param sequence: Input behavior sequence (e.g., [1, 2, 3, 4, 1])
    :param n: The n value for n-grams
    :return: distinct-n score
    """
    n_grams = [tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]

    # Count the occurrences of n-grams
    total_n_grams = len(n_grams)
    unique_n_grams = len(set(n_grams))

    # distinct-n score = number of unique n-grams / total number of n-grams
    if total_n_grams == 0:
        return 0
    return unique_n_grams / total_n_grams

def JSD(P, Q):   #JSD
    """Calculate Jensen-Shannon Divergence"""
    M = 0.5 * (P + Q)
    return 0.5 * kl_divergence(P, M) + 0.5 * kl_divergence(Q, M)

from scipy.stats import ks_2samp

def ks_test(real_data, generated_data):
    ks_statistic, p_value = ks_2samp(real_data, generated_data)
    return ks_statistic, p_value


from scipy.stats import wasserstein_distance

def wasserstein_distance_metric(real_data, generated_data):
    return wasserstein_distance(real_data, generated_data)



import numpy as np
from scipy.stats import entropy
from collections import Counter
def js_divergence(true_seq, gen_seq):
    # Distribution similarity
    true_counts = Counter(true_seq)
    gen_counts = Counter(gen_seq)
    all_acts = set(true_counts) | set(gen_counts)
    
    # JSD calculation
    P = np.array([true_counts.get(a,0) for a in all_acts]) + 1e-10
    Q = np.array([gen_counts.get(a,0) for a in all_acts]) + 1e-10
    M = 0.5 * (P + Q)
    jsd = 0.5 * (entropy(P, M) + entropy(Q, M))

    return jsd




import numpy as np
from scipy.stats import chisquare


def get_chisquare(true_seq, gen_seq,epsilon=1e-12):

    # Get all possible action categories
    all_actions = list(set(true_seq) | set(gen_seq))
    
    # Convert to frequency distribution
    true_counts = Counter(true_seq)
    gen_counts = Counter(gen_seq)
    
    # Create probability vectors
    P = np.array([true_counts.get(a, 0) for a in all_actions], dtype=float)
    Q = np.array([gen_counts.get(a, 0) for a in all_actions], dtype=float)
    
    # Add smoothing
    P += epsilon
    Q += epsilon
    
    # Normalization
    P /= P.sum()
    Q /= Q.sum()
    
    metrics ={}
    
    # 2. Total Variation Distance
    metrics['TVD'] = 0.5 * np.abs(P - Q).sum()
    # print(Q * len(gen_seq),P * len(true_seq))
    # 3. Chi-square test
    chi2_stat, p_value = chisquare(f_obs=Q , 
                                  f_exp=P )
    metrics['Chi2'] = {'statistic': chi2_stat, 'p-value': p_value}

    return metrics['TVD'],metrics['Chi2']['statistic'],metrics['Chi2']['p-value']


def get_tvd(real_data,generated_data):
    generated_counts = np.bincount(generated_data, minlength=len(index2id_final)+1) 

    # Frequency statistics of real data
    real_counts = np.bincount(real_data, minlength=len(index2id_final)+1)

    # Step 3: Normalize to calculate the probability distribution
    generated_prob = generated_counts / np.sum(generated_counts)
    real_prob = real_counts / np.sum(real_counts)
    
    # Step 4: Calculate the Total Variation Distance
    tvd = 0.5 * np.sum(np.abs(generated_prob - real_prob))
    
    # print(f"Total Variation Distance: {tvd}")
    return tvd


# imy_idx = 0
# num_start = 1000 * imy_idx
# num_end = 1000 * imy_idx +1000
# final_results =final_results[num_start:num_end]
# print(len(final_results))

"time"
# label_data = []
# pred_data = []
# for item in final_results:
#     label_data.append(item["time_label"])
# for item in final_results:
#     pred_data.append(item["time_pred"])



"loc"
label_data = []
pred_data = []
for item in final_results:
    label_data.append(item["loc_label"])
for item in final_results:
    pred_data.append(item["loc_pred"])

if 1:
    "Generated Data Evaluation"
    if 1:
        if 1:

            real_data = label_data
            gen_data = pred_data

            ks = []
            wd = []
            kl = []
            ad = []
            jsd = []
            bleu = []
            dn = []
            pv = []
            cs = []
            c_pv = []
            tvd = []
            tvd_ds = []

            final_real_data = []
            final_gen_data = []
            for i in range(len(gen_data)):
  
                # if len(real_data[i])<10:
                #     print('no',i)
                #     continue


                final_real_data += list(real_data[i])
                final_gen_data +=list(gen_data[i])
                continue

            final_real_data  =np.array(final_real_data)
            final_gen_data  =np.array(final_gen_data)
            # final_gen_data[final_gen_data<0] = 0
            print("Proportion of out-of-range data",np.sum(final_gen_data==-1)/final_gen_data.shape[0])
            final_gen_data =  final_gen_data[final_gen_data>=0]
            


            ks_statistic, p_value = ks_test(final_real_data, final_gen_data)
            ks.append(ks_statistic)
            pv.append(p_value)
            wd.append(wasserstein_distance_metric(final_real_data, final_gen_data))
            tvd_ds1,chi2_stat, p_value = get_chisquare(final_real_data, final_gen_data)
            cs.append(chi2_stat)
            c_pv.append(p_value)
            tvd_ds.append(tvd_ds1)
            tvd.append(get_tvd(final_real_data, final_gen_data))

            jsd.append(js_divergence(final_real_data, final_gen_data))
            bleu.append(bleu_score(final_real_data, final_gen_data))
            dn.append(distinct_n_score(final_real_data))

            print( "bleu:",np.mean(bleu), "tvd:",np.mean(tvd),"jsd:",np.mean(jsd))


import matplotlib.pyplot as plt
import numpy as np

# Data preparation (keeping original logic)

fintune_model = np.array(final_gen_data)
fintune_model  = fintune_model[final_gen_data<=18934]

label_data = np.array(final_real_data)
label_data = label_data[label_data<=18934]
# # # # Set histogram parameters ---------------------
bins = np.arange(0, 48) - 0.5  # Keep bins aligned with integers
plt.figure(figsize=(10, 8))

# Plot frequency histogram (add density=True)
plt.hist(fintune_model, bins=bins, alpha=0.5, 
         label='pred', color='blue', edgecolor='black',
         density=True)  # <-- Core modification

plt.hist(label_data, bins=bins, alpha=0.5,
         label='label', color='red', edgecolor='black',
         density=True)  # <-- Core modification

# Set plot attributes
plt.legend(loc='upper right')
plt.title('Probability Density Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')  # Changed to probability density

plt.tight_layout()
plt.savefig('frequency_distribution_8b_comparison_time.png')
plt.show()



"Zipf's law, vertical axis is frequency"
import collections
import matplotlib.pyplot as plt
import numpy as np

# Example data (replace with your own lists)
true_items = label_data
generated_items = fintune_model

print('label_data.shape,fintune_model.shape',label_data.shape,fintune_model.shape)

def get_sorted_frequencies(items):
    counter = collections.Counter(items)
    sorted_freqs = sorted(counter.values(), reverse=True)
    return np.arange(1, len(sorted_freqs) + 1), sorted_freqs

# Get ranks and frequencies
rank_true, freq_true = get_sorted_frequencies(true_items)
rank_gen, freq_gen = get_sorted_frequencies(generated_items)

# 📈 Plotting
plt.figure(figsize=(12, 5))

# --- 1. Linear Scale Plot ---
plt.subplot(1, 2, 1)
plt.plot(rank_true, freq_true, marker='o', label="True locs")
plt.plot(rank_gen, freq_gen, marker='x', label="Generated locs")
plt.title("Zipf-like Curve (Linear Scale)")
plt.xlabel("Rank")
plt.ylabel("Frequency")
plt.legend()

# --- 2. Log-Log Scale Plot ---
plt.subplot(1, 2, 2)
plt.loglog(rank_true, freq_true, marker='o', label="True locs")
plt.loglog(rank_gen, freq_gen, marker='x', label="Generated locs")
plt.title("Zipf-like Curve (Log-Log Scale)")
plt.xlabel("log(Rank)")
plt.ylabel("log(Frequency)")
plt.legend()

plt.tight_layout()
plt.savefig('Zipf-like Curve (Log-Log Scale).png')
plt.show()