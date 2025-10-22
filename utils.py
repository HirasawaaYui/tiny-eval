import subprocess

auto_command = """
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME="./datasets/"
export HF_ENDPOINT=https://hf-mirror.com

echo "Start evaluation"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "HF_HOME: $HF_HOME"
echo "HF_ENDPOINT: $HF_ENDPOINT"

accelerate launch \\
    --num_processes=8 \\
    --num_machines=1 \\
    --main_process_port=29502 \\
    -m \\
    lm_eval --model HFLM \\
    --model_args pretrained={model},tokenizer={model},trust_remote_code=True \\
    --tasks {tasks} \\
    --log_samples \\
    --output_path results/{proj_name}
    {batch_size} \\
    {meta_data} \\
"""

def launch_eval(model, tasks, proj_name, batch_size="", meta_data=""):
    command = auto_command.format(
        model=model,
        tasks=",".join(tasks),
        proj_name=proj_name,
        batch_size=f"--batch_size {batch_size}" if batch_size else "",
        meta_data=meta_data
    )

    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print('=' * 20)
        print(f"An error occurred while executing the command: {e}")
        print(command)
        print('=' * 20)