# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py

import random
import re
import string


import json
import re
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from tqdm import tqdm
# from prompt_mutl_task import *
import numpy as np

import re
import math
from collections import Counter



def get_answer(input_string):
    # Use regular expressions to extract the content between <answer> and </answer>
    match = re.search(r'<answer>(.*?)</answer>', input_string, re.DOTALL)

    if match:
        answer_content = match.group(1)  # extract the content between <answer> and </answer>
        # print(answer_content)
        return answer_content
    else:
        # print("Content between <answer> tags not found")
        return "00"
    
def extract_time_and_location_rl(text):
    pattern = r'''
        (?:-?\s*(\d{2}:\d{2})\s*[^\w<>]*<a_\d+><b_\d+><c_\d+><d_\d+>)  # List item format
        |                                                              # or
        (?:[Aa]t\s+(\d{2}:\d{2})\s+visited\s+location\s+<a_\d+><b_\d+><c_\d+><d_\d+>)  # At/at format
    '''
    
    # Find all matches, use VERBOSE for readability, IGNORECASE to ignore case
    matches = re.findall(pattern, text, re.VERBOSE | re.IGNORECASE)
    
    times = []
    locations = []
    
    # Process the match results
    for match in matches:
        # Extract time (the two formats are in different groups)
        time = match[0] if match[0] else match[1]
        
        if time:
            # Extract the corresponding location
            loc_pattern = r'<a_\d+><b_\d+><c_\d+><d_\d+>'
            # Find the location in the context containing the current time
            context = re.search(rf'{re.escape(time)}.*?{loc_pattern}', text, re.IGNORECASE)
            if context:
                loc_match = re.search(loc_pattern, context.group())
                if loc_match:
                    times.append(time)
                    locations.append(loc_match.group())
    
    return times, locations



from collections import Counter, defaultdict
# Given a string [time, location], output statistical features, all returned in dictionary form
def analyze_time_location(data):
    """
    Input: data = [(time_string, location_string), ...]
    Output: 
      1. Number of data points for each time period
      2. Top 3 locations with occurrences > 1
      3. Location occurrences for each time period (only keeping counts > 1)
    """

    # Define time periods
    time_ranges = {
        "early_morning": (0, 6),
        "morning": (6, 11),
        "noon": (11, 14),
        "afternoon": (14, 18),
        "evening": (18, 24)
    }

    # ---------- 1. Calculate the number of data points for each time period ----------
    time_period_count = defaultdict(int)
    for t, _ in data:
        hour = int(t.split(":")[0])
        for period, (start, end) in time_ranges.items():
            if start <= hour < end:
                time_period_count[period] += 1
                break

    # ---------- 2. Calculate location occurrences ----------
    location_counter = Counter(loc for _, loc in data)
    top_locations = {k: v for k, v in location_counter.most_common(3) if v > 1}

    # ---------- 3. Location occurrences for each time period (whole locations, not split) ----------
    time_period_locations = {period: Counter() for period in time_ranges.keys()}
    for t, loc in data:
        hour = int(t.split(":")[0])
        for period, (start, end) in time_ranges.items():
            if start <= hour < end:
                time_period_locations[period][loc] += 1
                break

    # Filter out locations with occurrences <= 1
    time_period_locations_filtered = {
        period: {k: v for k, v in counter.items() if v > 1}
        for period, counter in time_period_locations.items()
    }

    return dict(time_period_count), top_locations, time_period_locations_filtered



import re
from typing import List, Tuple
# Extract time and location pairs from the ground truth string
def parse_time_location(text: str) -> List[Tuple[str, str]]:
    """
    From a string like:
    'On Thursday: at 08:30 visited location <a_87><b_365><c_71><d_328>; at 09:00 visited location <...>; ... .'
    extract [("HH:MM", "<a_..><b_..><c_..><d_..>"), ...]
    """
    # Regex explanation:
    # - at\s+(\d{2}:\d{2}) captures the time
    # - \s+visited\s+location\s+ matches the fixed phrase (case-insensitive)
    # - ([^;.\n]+) captures the location until a semicolon/period/newline (not including them)
    pattern = re.compile(
        r"at\s+(\d{2}:\d{2})\s+visited\s+location\s+([^;.\n]+)",
        flags=re.IGNORECASE
    )

    results = []
    for time_str, loc in pattern.findall(text):
        # Strip whitespace from the location and keep the original entire string (not split)
        loc_clean = loc.strip()
        results.append((time_str, loc_clean))
    return results

def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=0.0,data_source='',extra_info=''):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = get_answer(solution_str)

    do_print = random.randint(1, 64) == 1
    # ground_truth = json.loads(ground_truth)
    
    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth}")
        if answer is not None:
            print(f"Extracted answer is not None: answer: {answer}\n")
        else:
            print("Extracted answer: None!",answer)
        # print(f"Solution string: {solution_str}")


    if 'Final Modified User Trajectory' in answer:
        time_pred, loc_pred = extract_time_and_location_rl(answer.split('Final Modified User Trajectory')[1])
    elif 'Final User Trajectory' in answer:
        time_pred, loc_pred = extract_time_and_location_rl(answer.split('Final User Trajectory')[1])
    else:
        return score


    label_data = parse_time_location(ground_truth)
    label_fea = analyze_time_location(label_data)


    pred_data = list(zip(time_pred, loc_pred))
    pred_fea = analyze_time_location(pred_data)

    if len(time_pred) != len(label_data):
        return score  
    
    # Check if the generated length is the same as the label length
    if len(label_data) != len(label_data):
        return (3- abs(len(label_data) - len(label_data))/len(label_data) * 3) + score  
     
    score +=3  

    # Calculate features item by item
    if pred_fea[0] == label_fea[0]:
        score +=1  

    format_list= ['early_morning', 'morning', 'noon', 'afternoon', 'evening']

    for _ in format_list:
        if set(pred_fea[2][_].keys()) == set(label_fea[2][_].keys()):
            score += 1
    print('final score:',score)
    return score