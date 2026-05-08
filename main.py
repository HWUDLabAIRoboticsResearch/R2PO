import argparse
import importlib

import gym_maze
import yaml
from dotenv import load_dotenv
from envs import nim, pong
from runner import llm_num_optim_runner
from runner import llm_num_optim_semantics_runner


load_dotenv()


ABLATION_RUNNERS = {
    "dist_state_reptraj": "ablations.reptraj.runner",
    "cont_state_reptraj": "ablations.reptraj.runner",
    "dist_state_three_traj": "ablations.three_traj.runner",
    "cont_state_three_traj": "ablations.three_traj.runner",
    "dist_state_reflective_prompted_policy_optimization": "ablations.reflective_prompted_policy_optimization.runner",
    "cont_state_reflective_prompted_policy_optimization": "ablations.reflective_prompted_policy_optimization.runner",
    "dist_state_always_critic": "ablations.always_critic.runner",
    "cont_state_always_critic": "ablations.always_critic.runner",
    "dist_state_critic_only": "ablations.critic_only.runner",
    "cont_state_critic_only": "ablations.critic_only.runner",
    "dist_state_actor_second_pass": "ablations.actor_second_pass.runner",
    "cont_state_actor_second_pass": "ablations.actor_second_pass.runner",
    "dist_state_pure_search": "ablations.pure_search.runner",
    "cont_state_pure_search": "ablations.pure_search.runner",
    "dist_state_tau_c_sensitivity": "ablations.tau_c_sensitivity.runner",
    "cont_state_tau_c_sensitivity": "ablations.tau_c_sensitivity.runner",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the config file",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    task = config["task"]
    if task in [
        "cont_space_llm_num_optim",
        "cont_space_llm_num_optim_rndm_proj",
        "dist_state_llm_num_optim",
    ]:
        llm_num_optim_runner.run_training_loop(**config)
    elif task in [
        "dist_state_llm_num_optim_semantics",
        "cont_state_llm_num_optim_semantics",
    ]:
        llm_num_optim_semantics_runner.run_training_loop(**config)
    elif task in ABLATION_RUNNERS:
        runner = importlib.import_module(ABLATION_RUNNERS[task])
        runner.run_training_loop(**config)
    else:
        raise ValueError(f"Task {task} not recognized.")


if __name__ == "__main__":
    main()
