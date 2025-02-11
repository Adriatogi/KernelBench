# python3 scripts/generate_samples.py run_name=gpt-4o-mini-2024-07-18_level_1 dataset_src=huggingface level=1 num_workers=50 server_type=openai model_name=gpt-4o-mini-2024-07-18
#python3 scripts/generate_samples_rerun.py run_name=gpt-4o-mini-2024-07-18_level_1_regenerate dataset_src=huggingface level=1 num_workers=1 server_type=openai model_name=gpt-4o-mini-2024-07-18
# python3 scripts/eval_from_generations.py level=1 run_name=gpt-4o-mini-2024-07-18_level_1 dataset_src="local" level="1" num_gpu_devices=4 timeout=300    
# python3 scripts/eval_from_generations.py level=1 run_name=gpt-4o-mini-2024-07-18_level_1_regenerate dataset_src="local" level="1" num_gpu_devices=4 timeout=300

# python3 scripts/generate_samples.py run_name=gpt-4-turbo-2024-04-09_level_1 dataset_src=huggingface level=1 num_workers=50 server_type=openai model_name=gpt-4-turbo-2024-04-09
# python3 scripts/generate_samples_rerun.py run_name=gpt-4-turbo-2024-04-09_level_1_regenerate dataset_src=huggingface level=1 num_workers=1 server_type=openai model_name=gpt-4-turbo-2024-04-09
# python3 scripts/eval_from_generations.py level=1 run_name=gpt-4-turbo-2024-04-09_level_1 dataset_src="local" level="1" num_gpu_devices=4 timeout=300
# python3 scripts/eval_from_generations.py level=1 run_name=gpt-4-turbo-2024-04-09_level_1_regenerate dataset_src="local" level="1" num_gpu_devices=4 timeout=300

# done
# #python3 scripts/generate_samples.py run_name=gpt-4o-2024-08-06_level_1 dataset_src=huggingface level=1 num_workers=10 server_type=openai model_name=gpt-4o-2024-08-06
# #python3 scripts/generate_samples_rerun.py run_name=gpt-4o-2024-08-06_level_1_regenerate dataset_src=huggingface level=1 num_workers=1 server_type=openai model_name=gpt-4o-2024-08-06
# #python3 scripts/eval_from_generations.py level=1 run_name=gpt-4o-2024-08-06_level_1 dataset_src="local" level="1" num_gpu_devices=4 timeout=300
# python3 scripts/eval_from_generations.py level=1 run_name=gpt-4o-2024-08-06_level_1_regenerate dataset_src="local" level="1" num_gpu_devices=4 timeout=600

# python3 scripts/generate_samples.py run_name=meta-llama-3.1-405b-instruct-turbo_level_1 dataset_src=huggingface level=1 num_workers=1 server_type=together model_name=meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo
# python3 scripts/generate_samples_rerun.py run_name=meta-llama-3.1-405b-instruct-turbo_level_1_regenerate dataset_src=huggingface level=1 num_workers=1 server_type=together model_name=meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo
# python3 scripts/eval_from_generations.py level=1 run_name=meta-llama-3.1-405b-instruct-turbo_level_1 dataset_src="local" level="1" num_gpu_devices=4 timeout=300
# python3 scripts/eval_from_generations.py level=1 run_name=meta-llama-3.1-405b-instruct-turbo_level_1_regenerate dataset_src="local" level="1" num_gpu_devices=4 timeout=300

# python3 scripts/generate_samples.py run_name=claude-3-5-sonnet-20240620_level_1 dataset_src=huggingface level=1 num_workers=1 server_type=anthropic model_name=claude-3-5-sonnet-20240620
# python3 scripts/generate_samples_rerun.py run_name=claude-3-5-sonnet-20240620_level_1_regenerate dataset_src=huggingface level=1 num_workers=1 server_type=anthropic model_name=claude-3-5-sonnet-20240620
# python3 scripts/eval_from_generations.py level=1 run_name=claude-3-5-sonnet-20240620_level_1 dataset_src="local" level="1" num_gpu_devices=4 timeout=300
# python3 scripts/eval_from_generations.py level=1 run_name=claude-3-5-sonnet-20240620_level_1_regenerate dataset_src="local" level="1" num_gpu_devices=4 timeout=300


#python3 scripts/generate_samples.py run_name=claude-3-5-sonnet-20240620_level_1 dataset_src=huggingface level=1 num_workers=1 server_type=anthropic model_name=claude-3-5-sonnet-20240620
#python3 scripts/generate_samples.py run_name=o1-2024-12-17_level_1 dataset_src=huggingface level=1 num_workers=50 server_type=openai model_name=o1-2024-12-17
#python3 scripts/generate_samples.py run_name=o1-mini-2024-09-12_level_1 dataset_src=huggingface level=1 num_workers=50 server_type=openai model_name=o1-mini-2024-09-12

# done python3 scripts/eval_from_generations.py level=1 run_name=gpt-4-turbo-2024-04-09_level_1 dataset_src="local" level="1" num_gpu_devices=4 timeout=300
#python3 scripts/eval_from_generations.py level=1 run_name=claude-3-5-sonnet-20240620_level_1 dataset_src="local" level="1" num_gpu_devices=4 timeout=300
#python3 scripts/eval_from_generations.py level=1 run_name=o1-2024-12-17_level_1 dataset_src="local" level="1" num_gpu_devices=4 timeout=300
#python3 scripts/eval_from_generations.py level=1 run_name=o1-mini-2024-09-12_level_1 dataset_src="local" level="1" num_gpu_devices=4 timeout=300

# python3 scripts/generate_samples.py run_name=archon-multi-model-gpt4-critic-gpt4-ranker-gpt4-fusion_level_1 dataset_src=huggingface level=1 num_workers=50 server_type=archon model_name=archon-multi-model-gpt4-critic-gpt4-ranker-gpt4-fusion archon_config_path=archon_configs/archon-multi-model-gpt4-critic-gpt4-ranker-gpt4-fusion.json
# python3 scripts/eval_from_generations.py level=1 run_name=archon-multi-model-gpt4-critic-gpt4-ranker-gpt4-fusion_level_1 dataset_src="local" level="1" num_gpu_devices=4 timeout=300

# done python3 scripts/kernel_seq.py -l 1 --llms claude-sonnet -r claude1
# done python3 scripts/kernel_seq.py -l 1 --llms llama -r llama1 -s 69
# done python3 scripts/kernel_seq.py -l 1 --llms claude-sonnet claude-sonnet claude-sonnet claude-sonnet claude-sonnet -r claude5 -s 82
# done python3 scripts/kernel_seq.py -l 1 --llms claude-sonnet claude-sonnet claude-sonnet claude-sonnet claude-sonnet claude-sonnet claude-sonnet claude-sonnet claude-sonnet claude-sonnet -r claude10 -s 85
# python3 scripts/kernel_seq.py -l 1 --llms llama llama llama llama llama -r llama5 -s 73 -n 2
# python3 scripts/kernel_seq.py -l 1 --llms llama llama llama -r llama3 -s 58 -n 2
# python3 scripts/kernel_seq.py -l 1 --llms llama llama llama llama claude-sonnet claude-sonnet claude-sonnet -r llama4_claude3 -s 48 -n 2

# python3 scripts/eval_from_generations.py level=1 run_name=claude1 dataset_src="local" level="1" num_gpu_devices=4 timeout=300
# python3 scripts/eval_from_generations.py level=1 run_name=llama1 dataset_src="local" level="1" num_gpu_devices=4 timeout=300
# python3 scripts/eval_from_generations.py level=1 run_name=llama3 dataset_src="local" level="1" num_gpu_devices=4 timeout=300
# python3 scripts/eval_from_generations.py level=1 run_name=llama4_claude3 dataset_src="local" level="1" num_gpu_devices=4 timeout=300
# python3 scripts/eval_from_generations.py level=1 run_name=llama5 dataset_src="local" level="1" num_gpu_devices=4 timeout=300
# python3 scripts/eval_from_generations.py level=1 run_name=claude5 dataset_src="local" level="1" num_gpu_devices=4 timeout=300
# python3 scripts/eval_from_generations.py level=1 run_name=claude10 dataset_src="local" level="1" num_gpu_devices=4 timeout=300


#python3 scripts/kernel_seq.py -l 1 --llms claude-sonnet claude-sonnet claude-sonnet -r claude3 -n 2
#python3 scripts/kernel_seq.py -l 1 --llms llama claude-sonnet claude-sonnet claude-sonnet -r llama1_claude3 -n 2
#python3 scripts/kernel_seq.py -l 1 --llms claude-sonnet llama llama llama -r claude1_llama3 -n 2

# python3 scripts/eval_from_generations.py level=1 run_name=claude1_llama3 dataset_src="local" level="1" num_gpu_devices=4 timeout=300
# python3 scripts/eval_from_generations.py level=1 run_name=llama1_claude3 dataset_src="local" level="1" num_gpu_devices=4 timeout=300
# python3 scripts/eval_from_generations.py level=1 run_name=claude3 dataset_src="local" level="1" num_gpu_devices=4 timeout=300

# python3 scripts/kernel_seq.py -l 1 --llms gpt4o -r gpt4o1 -n 2
# python3 scripts/kernel_seq.py -l 1 --llms gpt4o gpt4o gpt4o gpt4o gpt4o -r gpt4o5 -n 2
# python3 scripts/kernel_seq.py -l 1 --llms gpt4o gpt4o gpt4o gpt4o gpt4o gpt4o gpt4o gpt4o gpt4o gpt4o -r gpt4o10 -n 2

# python3 scripts/eval_from_generations.py level=1 run_name=gpt4o1 dataset_src="local" level="1" num_gpu_devices=4 timeout=300
# python3 scripts/eval_from_generations.py level=1 run_name=gpt4o5 dataset_src="local" level="1" num_gpu_devices=4 timeout=300
# python3 scripts/eval_from_generations.py level=1 run_name=gpt4o10 dataset_src="local" level="1" num_gpu_devices=4 timeout=300

# python3 scripts/kernel_seq.py -l 2 --llms claude-sonnet -r claude1 -n 2
# python3 scripts/kernel_seq.py -l 2 --llms claude-sonnet claude-sonnet claude-sonnet claude-sonnet claude-sonnet -r claude5 -n 2
# python3 scripts/eval_from_generations.py level=2 run_name=claude1_level_2 dataset_src="local" level="2" num_gpu_devices=4 timeout=300
# python3 scripts/eval_from_generations.py level=2 run_name=claude5_level_2 dataset_src="local" level="2" num_gpu_devices=4 timeout=300

# python3 scripts/kernel_seq.py -l 2 --llms claude-sonnet claude-sonnet claude-sonnet claude-sonnet claude-sonnet claude-sonnet claude-sonnet claude-sonnet claude-sonnet claude-sonnet -r claude10_level_2 -n 2 -s 77
# python3 scripts/eval_from_generations.py level=2 run_name=claude10_level_2 dataset_src="local" level="2" num_gpu_devices=4 timeout=300

# done python3 scripts/kernel_seq.py -l 2 --llms llama -r llama1_level_2 -n 2
# done python3 scripts/kernel_seq.py -l 2 --llms llama llama llama llama llama -r llama5_level_2 -n 2
# python3 scripts/kernel_seq.py -l 2 --llms llama llama llama llama llama llama llama llama llama llama llama -r llama10_level_2 -n 2 -s 36

# python3 scripts/eval_from_generations.py level=2 run_name=llama1_level_2 dataset_src="local" level="2" num_gpu_devices=4 timeout=300
# python3 scripts/eval_from_generations.py level=2 run_name=llama5_level_2 dataset_src="local" level="2" num_gpu_devices=4 timeout=300
# python3 scripts/eval_from_generations.py level=2 run_name=llama10_level_2 dataset_src="local" level="2" num_gpu_devices=4 timeout=300

# done python3 scripts/kernel_seq.py -l 2 --llms gpt4o -r gpt4o1_level_2 -n 2
# done python3 scripts/eval_from_generations.py level=2 run_name=gpt4o1_level_2 dataset_src="local" level="2" num_gpu_devices=4 timeout=300

# python3 scripts/kernel_seq.py -l 2 --llms gpt4o gpt4o gpt4o gpt4o gpt4o -r gpt4o5_level_2 -n 2
# python3 scripts/eval_from_generations.py level=2 run_name=gpt4o5_level_2 dataset_src="local" level="2" num_gpu_devices=4 timeout=300
# python3 scripts/kernel_seq.py -l 2 --llms gpt4o gpt4o gpt4o gpt4o gpt4o gpt4o gpt4o gpt4o gpt4o gpt4o -r gpt4o10_level_2 -n 2
# python3 scripts/eval_from_generations.py level=2 run_name=gpt4o10_level_2 dataset_src="local" level="2" num_gpu_devices=4 timeout=300

# python3 scripts/kernel_seq.py -l 1 --llms r1 -r r1_level_1 -n 1 -g 1
# python3 scripts/eval_from_generations.py level=1 run_name=r1_level_1 dataset_src="local" level="1" num_gpu_devices=4 timeout=300


# python3 scripts/kernel_seq_copy.py -l 1 -r gpt4o3_claude3_level_1 --llms gpt4o gpt4o gpt4o claude-sonnet claude-sonnet claude-sonnet -n 2
# python3 scripts/kernel_seq_copy.py -l 1 -r claude3_gpt4o3_level_1 --llms claude-sonnet claude-sonnet claude-sonnet gpt4o gpt4o gpt4o  -n 2
# python3 scripts/kernel_seq_copy.py -l 1 -r gpt4o1_claude_3_level_1 --llms gpt4o claude-sonnet claude-sonnet claude-sonnet -n 2
#python3 scripts/kernel_seq_copy.py -l 2 -r claude1_level_2 --llms claude-sonnet -n 2

# python3 scripts/eval_from_generations.py level=1 run_name=gpt4o3_claude3_level_1 dataset_src="local" level="1" num_gpu_devices=4 timeout=300
# python3 scripts/eval_from_generations.py level=1 run_name=claude3_gpt4o3_level_1 dataset_src="local" level="1" num_gpu_devices=4 timeout=300
# python3 scripts/eval_from_generations.py level=1 run_name=gpt4o1_claude_3_level_1 dataset_src="local" level="1" num_gpu_devices=4 timeout=300
#python3 scripts/eval_from_generations.py level=2 run_name=claude1_level_2 dataset_src="local" level="2" num_gpu_devices=4 timeout=300


# done python3 scripts/kernel_seq_copy.py -l 3 --llms claude-sonnet -r claude1_level_3 -n 2
# done python3 scripts/kernel_seq_copy.py -l 3 --llms claude-sonnet claude-sonnet claude-sonnet -r claude3_level_3 -n 2
# done python3 scripts/kernel_seq_copy.py -l 3 --llms claude-sonnet claude-sonnet claude-sonnet claude-sonnet claude-sonnet -r claude5_level_3 -n 2
# done python3 scripts/kernel_seq_copy.py -l 3 --llms claude-sonnet claude-sonnet claude-sonnet claude-sonnet claude-sonnet claude-sonnet claude-sonnet claude-sonnet claude-sonnet claude-sonnet -r claude10_level_3 -n 2

# done python3 scripts/kernel_seq_copy.py -l 3 --llms llama -r llama1_level_3 -n 2
# done python3 scripts/kernel_seq_copy.py -l 3 --llms llama llama llama -r llama3_level_3 -n 2
# done python3 scripts/kernel_seq_copy.py -l 3 --llms llama llama llama llama llama -r llama5_level_3 -n 2
# done python3 scripts/kernel_seq_copy.py -l 3 --llms llama llama llama llama claude-sonnet claude-sonnet claude-sonnet -r llama4_claude3_level_3 -n 2
# done python3 scripts/kernel_seq_copy.py -l 3 --llms claude-sonnet llama llama llama -r claude1_llama3_level_3 -n 2
# done python3 scripts/kernel_seq_copy.py -l 3 --llms llama claude-sonnet claude-sonnet claude-sonnet -r llama1_claude3_level_3 -n 2

# done python3 scripts/kernel_seq_copy.py -l 3 --llms gpt4o -r gpt4o1_level_3 -n 2


# python3 scripts/eval_from_generations.py level=3 run_name=claude1_level_3 dataset_src="local" level="3" num_gpu_devices=4 timeout=300
# python3 scripts/eval_from_generations.py level=3 run_name=claude3_level_3 dataset_src="local" level="3" num_gpu_devices=4 timeout=300
# python3 scripts/eval_from_generations.py level=3 run_name=claude5_level_3 dataset_src="local" level="3" num_gpu_devices=4 timeout=300
# python3 scripts/eval_from_generations.py level=3 run_name=claude10_level_3 dataset_src="local" level="3" num_gpu_devices=4 timeout=300

# python3 scripts/eval_from_generations.py level=3 run_name=llama1_level_3 dataset_src="local" level="3" num_gpu_devices=4 timeout=300
# python3 scripts/eval_from_generations.py level=3 run_name=llama3_level_3 dataset_src="local" level="3" num_gpu_devices=4 timeout=300
# python3 scripts/eval_from_generations.py level=3 run_name=llama5_level_3 dataset_src="local" level="3" num_gpu_devices=4 timeout=300
# python3 scripts/eval_from_generations.py level=3 run_name=llama4_claude3_level_3 dataset_src="local" level="3" num_gpu_devices=4 timeout=300
# python3 scripts/eval_from_generations.py level=3 run_name=claude1_llama3_level_3 dataset_src="local" level="3" num_gpu_devices=4 timeout=300
# python3 scripts/eval_from_generations.py level=3 run_name=llama1_claude3_level_3 dataset_src="local" level="3" num_gpu_devices=4 timeout=300

# python3 scripts/eval_from_generations.py level=3 run_name=gpt4o1_level_3 dataset_src="local" level="3" num_gpu_devices=4 timeout=300

# python3 scripts/kernel_seq_copy.py -l 3 --llms gpt4o gpt4o gpt4o gpt4o gpt4o -r gpt4o5_level_3 -n 2
# python3 scripts/kernel_seq_copy.py -l 3 --llms gpt4o gpt4o gpt4o gpt4o gpt4o gpt4o gpt4o gpt4o gpt4o gpt4o -r gpt4o10_level_3 -n 2
# python3 scripts/kernel_seq_copy.py -l 3 --llms gpt4o claude-sonnet claude-sonnet claude-sonnet -r gpt4o1_claude_3_level_3 -n 2 -g 3
# python3 scripts/kernel_seq_copy.py -l 3 --llms gpt4o gpt4o gpt4o claude-sonnet claude-sonnet claude-sonnet -r gpt4o3_claude3_level_3 -n 2 -g 3
# python3 scripts/kernel_seq_copy.py -l 3 --llms claude-sonnet claude-sonnet claude-sonnet gpt4o gpt4o gpt4o -r claude3_gpt4o3_level_3 -n 2 -g 3

# python3 scripts/eval_from_generations.py level=3 run_name=gpt4o5_level_3 dataset_src="local" level="3" num_gpu_devices=4 timeout=300
# python3 scripts/eval_from_generations.py level=3 run_name=gpt4o10_level_3 dataset_src="local" level="3" num_gpu_devices=4 timeout=300
#python3 scripts/eval_from_generations.py level=3 run_name=gpt4o3_claude3_level_3 dataset_src="local" level="3" num_gpu_devices=3 timeout=300
# python3 scripts/eval_from_generations.py level=3 run_name=claude3_gpt4o3_level_3 dataset_src="local" level="3" num_gpu_devices=3 timeout=300
# python3 scripts/eval_from_generations.py level=3 run_name=gpt4o1_claude_3_level_3 dataset_src="local" level="3" num_gpu_devices=3 timeout=300

