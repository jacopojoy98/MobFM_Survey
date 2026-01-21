import json
import numpy as np

def analyze_jsonl(file_path):
    total = 0
    correct = 0
    predict_set = set()
    label_set = set()



    label_in_history_correct = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            total +=1
            # if total>=500:
            #     continue
            data = json.loads(line)
            # total += 1
            label = data["label"].split(',')[0]

            id = int(data["prompt"])

            data["predict"] = data["predict"].split(' ')[-1].split('.')[0]
            if data["predict"] == label:
                correct += 1
            predict_set.add(data["predict"])
            label_set.add(label)

            # if total >=15000:
            #     break

    accuracy = correct / total if total > 0 else 0
    predict_size = len(predict_set)
    label_size = len(label_set)
    predict_repeat_rate = predict_size / total if total > 0 else 0
    label_repeat_rate = label_size / total if total > 0 else 0

    intersection = predict_set & label_set
    predict_only = predict_set - label_set
    label_only = label_set - predict_set

    print(correct,label_in_history_correct,"correct,label_in_history_correct")

    in_label_predict_num = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if data["predict"] in label_set:
                in_label_predict_num += 1


    result = {
        "total": total,
        "accuracy": accuracy,
        "label_in_history_accuracy": label_in_history_correct/total,
        "predict_set_size": predict_size,
        "label_set_size": label_size,
        "predict_repeat_rate": predict_repeat_rate,
        "label_repeat_rate": label_repeat_rate,
        "no_hallucination_rate": len(intersection) / predict_size if predict_size > 0 else 0,
        "in_label_accuracy":  correct/ in_label_predict_num if in_label_predict_num > 0 else 0,
    }
    return result


import json

def parse_tags(s):
    """turn '<a_213><b_22><c_241><d_312>'  to  ['a_213','b_22','c_241','d_312']"""
    return [x.strip("<>") for x in s.split(">") if x]

def compute_accuracy(jsonl_file):
    total = 0
    correct_total = 0
    
    prefix_correct = [0, 0, 0, 0]  
    pos_correct = [0, 0, 0, 0]    
    
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            label_tags = parse_tags(data["label"].split('is most likely to visit is: ')[0])
            try:
                pred_tags = parse_tags(data["predict"].split('is most likely to visit is: ')[0])
            except:
                # print(data["predict"])
                continue

            # label_tags = parse_tags(data["label"])
            # pred_tags = parse_tags(data["predict"])
            # print(label_tags)
            # print(pred_tags)
            total += 1

            if not (total>0 and total<=5000):
                continue
            
            try:
  
                if label_tags == pred_tags:
                    correct_total += 1

      
                for i in range(4):
                    if label_tags[:i+1] == pred_tags[:i+1]:
                        prefix_correct[i] += 1
                for i in range(4):
                    if label_tags[i] == pred_tags[i]:
                        pos_correct[i] += 1
            except:
                continue
    print(total)
    overall_acc = correct_total / total if total > 0 else 0
    prefix_acc = [c / total if total > 0 else 0 for c in prefix_correct]
    pos_acc = [c / total if total > 0 else 0 for c in pos_correct]
    
    return overall_acc, prefix_acc, pos_acc


if __name__ == "__main__":

    file_path = "eval_result/predict_ana_city/eval_tot.jsonl"  # Replace with your file path

    

    stats = analyze_jsonl(file_path)
    print(stats)

    overall_acc, prefix_acc, pos_acc = compute_accuracy(file_path)
    
    print(f"Overall Accuracy: {overall_acc:.4f}")
    for i, acc in enumerate(prefix_acc, start=1):
        print(f"Prefix {i} Accuracy: {acc:.4f}")
    for i, acc in enumerate(pos_acc, start=1):
        print(f"Position {i} Accuracy: {acc:.4f}")