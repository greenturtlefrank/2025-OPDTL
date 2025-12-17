torchrun --nproc_per_node 1 example_text_completion.py \
    --ckpt_dir /home/greenturtlefrank/.llama/checkpoints/Llama3.1-8B-Instruct \
    --tokenizer_path /home/greenturtlefrank/.llama/checkpoints/Llama3.1-8B-Instruct/tokenizer.model \
    --max_seq_len 1024 \
    --max_gen_len 512 \