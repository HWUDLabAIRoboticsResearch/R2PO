from world.continuous_space_general_world import ContinualSpaceGeneralWorld
from world.discrete_state_general_world import DiscreteStateGeneralWorld
from agent.llm_num_optim_linear_policy_rndm_proj import LLMNumOptimRndmPrjAgent
from agent.llm_num_optim_linear_policy import LLMNumOptimAgent
from agent.llm_num_optim_q_table import LLMNumOptimQTableAgent
from jinja2 import Environment, FileSystemLoader
import os
import re
import traceback
import numpy as np
from runner.resume_utils import resolve_run_logdir, restore_agent_from_run


def run_training_loop(
    task,
    num_episodes,
    gym_env_name,
    render_mode,
    logdir,
    dim_actions,
    dim_states,
    max_traj_count,
    max_traj_length,
    template_dir,
    llm_si_template_name,
    llm_output_conversion_template_name,
    llm_model_name,
    num_evaluation_episodes,
    warmup_episodes,
    warmup_dir,
    bias=None,
    rank=None,
    optimum=1000,
    search_step_size=0.1,
    env_kwargs=None,
    rerun=None,
    resume_run=None,
    policy_variant=None,
    hidden_dim=None,
    latent_dim_layer1=None,
    latent_dim_layer2=None,
    projection_seed=None,
    projection_seed_base=1000,
    activation="tanh",
    param_min=-6.0,
    param_max=6.0,
    latent_init_std=2.0,
    decoded_weight_scale=1.0,
):
    assert task in ["cont_space_llm_num_optim", "cont_space_llm_num_optim_rndm_proj", "dist_state_llm_num_optim"]

    logdir, run_mode = resolve_run_logdir(logdir, rerun=rerun, resume_run=resume_run)
    os.makedirs(logdir, exist_ok=True)

    resolved_projection_seed = projection_seed
    if task == "cont_space_llm_num_optim":
        resolved_policy_variant = policy_variant
        if resolved_policy_variant is None and hidden_dim is not None and latent_dim_layer1 is not None and latent_dim_layer2 is not None:
            resolved_policy_variant = "projected_mlp_qr"
        if resolved_policy_variant == "projected_mlp_qr" and resolved_projection_seed is None:
            run_name = os.path.basename(logdir.rstrip("/"))
            match = re.match(r"run_(\d+)$", run_name)
            run_idx = int(match.group(1)) if match else 1
            resolved_projection_seed = int(projection_seed_base) + run_idx
    else:
        resolved_policy_variant = policy_variant

    jinja2_env = Environment(loader=FileSystemLoader(template_dir))
    llm_si_template = jinja2_env.get_template(llm_si_template_name)
    llm_output_conversion_template = jinja2_env.get_template(
        llm_output_conversion_template_name
    )

    if task in ["cont_space_llm_num_optim", "cont_space_llm_num_optim_rndm_proj"]:
        world = ContinualSpaceGeneralWorld(
            gym_env_name,
            render_mode,
            max_traj_length,
        )


        if task == "cont_space_llm_num_optim":
            agent = LLMNumOptimAgent(
                logdir,
                dim_actions,
                dim_states,
                max_traj_count,
                max_traj_length,
                llm_si_template,
                llm_output_conversion_template,
                llm_model_name,
                num_evaluation_episodes,
                bias,
                optimum,
                search_step_size,
                policy_variant=resolved_policy_variant,
                hidden_dim=hidden_dim,
                latent_dim_layer1=latent_dim_layer1,
                latent_dim_layer2=latent_dim_layer2,
                projection_seed=resolved_projection_seed,
                activation=activation,
                param_min=param_min,
                param_max=param_max,
                latent_init_std=latent_init_std,
                decoded_weight_scale=decoded_weight_scale,
            )
        elif task == "cont_space_llm_num_optim_rndm_proj":
            agent = LLMNumOptimRndmPrjAgent(
                logdir,
                dim_actions,
                dim_states,
                max_traj_count,
                max_traj_length,
                llm_si_template,
                llm_output_conversion_template,
                llm_model_name,
                num_evaluation_episodes,
                rank,
                bias,
                optimum,
            )


    elif task == "dist_state_llm_num_optim":
        world = DiscreteStateGeneralWorld(
            gym_env_name,
            render_mode,
            max_traj_length,
            env_kwargs=env_kwargs,
        )

        agent = LLMNumOptimQTableAgent(
            logdir,
            dim_actions,
            dim_states,
            max_traj_count,
            max_traj_length,
            llm_si_template,
            llm_output_conversion_template,
            llm_model_name,
            num_evaluation_episodes,
            optimum,
            env_kwargs=env_kwargs,
        )

        print('init done')
    agent.max_iterations = num_episodes
    if run_mode == "resume":
        start_episode = restore_agent_from_run(
            agent,
            logdir,
            semantic=False,
            is_qtable=(task == "dist_state_llm_num_optim"),
            warmup_dir=warmup_dir,
        )
    else:
        start_episode = 0
        if not warmup_dir:
            warmup_dir = f"{logdir}/warmup"
            os.makedirs(warmup_dir, exist_ok=True)
            agent.random_warmup(world, warmup_dir, warmup_episodes)
        else:
            agent.replay_buffer.load(warmup_dir)

    overall_log_path = f"{logdir}/overall_log.txt"
    overall_log_mode = "a" if run_mode == "resume" and os.path.exists(overall_log_path) else "w"
    overall_log_file = open(overall_log_path, overall_log_mode)
    if overall_log_mode == "w":
        overall_log_file.write("Iteration, CPU Time, API Time, Total Episodes, Total Steps, Total Reward\n")
        overall_log_file.flush()

    if start_episode >= num_episodes:
        print(f"Run already has {start_episode} episodes; nothing to resume.")
        overall_log_file.close()
        return

    for episode in range(start_episode, num_episodes):
        print(f"Episode: {episode}")
        # create log dir
        curr_episode_dir = f"{logdir}/episode_{episode}"
        print(f"Creating log directory: {curr_episode_dir}")
        os.makedirs(curr_episode_dir, exist_ok=True)
        
        for trial_idx in range(5):
            try:
                cpu_time, api_time, total_episodes, total_steps, total_reward = agent.train_policy(world, curr_episode_dir)
                overall_log_file.write(f"{episode + 1}, {cpu_time}, {api_time}, {total_episodes}, {total_steps}, {total_reward}\n")
                overall_log_file.flush()
                print(f"{trial_idx + 1}th trial attempt succeeded in training")
                break
            except Exception as e:
                print(
                    f"{trial_idx + 1}th trial attempt failed with error in training: {e}"
                )
                traceback.print_exc()

                if trial_idx == 4:
                    print(f"All {trial_idx + 1} trials failed. Train terminated")
                    exit(1)
                continue
    overall_log_file.close()
