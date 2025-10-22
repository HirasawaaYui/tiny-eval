import sys, os

tiny_pretrainer_path = '../tiny-pretrainer/'
sys.path.append(tiny_pretrainer_path)

from tasks import *
from utils import launch_eval

models = [
    "results/FIM/baseline_gated_deltanet/350m_FIM/fineweb-baseline-8k_",
]

tasks = COMMONSENSE_TASKS

if __name__ == "__main__":
    models = [os.path.abspath(os.path.join(tiny_pretrainer_path, model)) for model in models]

    for path in models:
        proj_name = '_'.join(path.split('/')[-3])
        launch_eval(
            model=path,
            tasks=tasks,
            proj_name=proj_name,
        )
