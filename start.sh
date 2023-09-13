!#/bin/bash


# python tiles_train.py --model=EarlyJoinSAGE --toy_data=True
# # graph train&test
# python layout_train.py --epochs 1 --toy_data=True

# On xla:random
python layout_train.py --source xla --search random --epochs 10 --max_configs 2000  --configs 2 --keep_nodes -1

# On xla:default
python layout_train.py --source xla --search default --epochs 10 --max_configs 2000  --configs 2 --keep_nodes -1

# On nlp:random
python layout_train.py --source nlp --search random --epochs 10 --max_configs 2000  --configs 2 --keep_nodes -1

# On nlp:default
python layout_train.py --source nlp --search default --epochs 10 --max_configs 2000  --configs 2 --keep_nodes -1

# tile train&test
python tiles_train.py --model=EarlyJoinSAGE --epochs 20

python combine_csvs.py

MSG="init_version"
kaggle competitions submit predict-ai-model-runtime -f /root/out/tpugraphs_submission.csv -m "${MSG}"
