from dp_pipeline import *

lang_list=["English","Germany","Arabic","China"]
model_path_list = ["models/China/Llama-3.2-1B-Instruct-China-dp","models/Arabic/Llama-3.2-1B-Instruct-Arabic-dp"]
# "/home/qxy/models/Llama-3.2-1B-Instruct"
for lang in lang_list:
    for model_path in model_path_list:
        if lang in model_path:
            continue
        if lang=="Arabic":
            datapath = f"data/Arabic/Finetune/WVQ_arabic_Iraq_Jordan_llama.jsonl"
        else:
            datapath = f"data/{lang}/Finetune/WVQ_{lang}_llama.jsonl"
        dataset = load_dataset('json', data_files=datapath, split='train')
        splited = dataset.train_test_split(test_size=0.2,seed=42)
        # pref_dp_f1 = eval(model_path,0,splited['test'],metric="distance")
        # base_f1 = eval("/home/qxy/models/Llama-3.2-1B-Instruct",0,splited['test'],metric="distance")
        # # print(f"DP model F1: {pref_dp_f1}, Base model F1: {base_f1}")
        # print(f"DP model avg dist: {pref_dp_f1}, Base model avg dist: {base_f1}")


        weighted_f1 = eval(model_path,"cuda:3",splited['test'])[-1]
        avg_distance = eval(model_path,"cuda:3",splited['test'],metric="distance")
        # save to file
        if not os.path.exists("results/score.json"):
            with open("results/score.json", 'w') as fw:
                json.dump({}, fw)
                read_scores = {}
        else:
            with open(f"results/score.json", 'r') as fr:
                read_scores = json.load(fr)
        if model_path not in read_scores.keys():
            read_scores[model_path] = {}
        if lang not in read_scores[model_path].keys():
            read_scores[model_path][lang] = {}
        read_scores[model_path][lang]['weighted_f1'] = weighted_f1
        read_scores[model_path][lang]['avg_distance'] = avg_distance
        with open(f"results/score.json", 'w') as fw:
            json.dump(read_scores, fw, indent=4)