export CUDA_VISIBLE_DEVICES=0
export HF_HOME="./datasets/"
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=/home/test/test07/shenxingyu/tiny_pretrainer:$PYTHONPATH
export HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_DATASETS_OFFLINE=1

echo "Start evaluation"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "HF_HOME: $HF_HOME"
echo "HF_ENDPOINT: $HF_ENDPOINT"

accelerate launch \
    --num_processes=8 \
    --num_machines=1 \
    --main_process_port=29502 \
    -m \
    lm_eval --model hf \
    --model_args pretrained=/home/test/test07/shenxingyu/tiny_pretrainer/results/FIM/baseline_gated_deltanet/350m_FIM/fineweb-baseline-8k_,tokenizer=/home/test/test07/shenxingyu/tiny_pretrainer/results/FIM/baseline_gated_deltanet/350m_FIM/fineweb-baseline-8k_,trust_remote_code=True \
    --tasks wikitext,lambada_standard,piqa,hellaswag,winogrande,arc_easy,arc_challenge,social_iqa,boolq,swde,squadv2,fda,triviaqa,drop \
    --log_samples \
    --output_path results/baseline_gated_deltanet_350m_FIM_fineweb-baseline-8k_
     \
     \