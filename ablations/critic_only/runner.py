import os
import traceback

from ablations.critic_only.linear_semantics_agent import (
    ReflectiveCriticOnlyLinearSemanticsAgent,
)
from ablations.critic_only.q_table_semantics_agent import (
    ReflectiveCriticOnlyQTableSemanticsAgent,
)
from jinja2 import Environment, FileSystemLoader
from runner.resume_utils import resolve_run_logdir, restore_agent_from_run
from world.continuous_space_general_world import ContinualSpaceGeneralWorld
from world.discrete_state_general_world import DiscreteStateGeneralWorld


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
    critic_llm_template_name,
    bias=None,
    rank=None,
    optimum=1000,
    search_step_size=0.1,
    env_kwargs=None,
    env_desc_file=None,
    critic_llm_env_desc_file=None,
    rerun=None,
    resume_run=None,
    traj_history_last_n=None,
    **_extra,
):
    assert task in [
        "dist_state_critic_only",
        "cont_state_critic_only",
    ]

    logdir, run_mode = resolve_run_logdir(logdir, rerun=rerun, resume_run=resume_run)
    os.makedirs(logdir, exist_ok=True)

    jinja2_env = Environment(loader=FileSystemLoader([template_dir, "ablations"]))
    llm_si_template = jinja2_env.get_template(llm_si_template_name)
    llm_output_conversion_template = jinja2_env.get_template(
        llm_output_conversion_template_name
    )
    critic_llm_env_desc_file = critic_llm_env_desc_file or env_desc_file

    if task == "dist_state_critic_only":
        world = DiscreteStateGeneralWorld(
            gym_env_name,
            render_mode,
            max_traj_length,
            env_kwargs=env_kwargs,
        )
        agent = ReflectiveCriticOnlyQTableSemanticsAgent(
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
            env_desc_file=env_desc_file,
            jinja2_env=jinja2_env,
            critic_llm_template_name=critic_llm_template_name,
            critic_llm_env_desc_file=critic_llm_env_desc_file,
        )
    else:
        world = ContinualSpaceGeneralWorld(
            gym_env_name,
            render_mode,
            max_traj_length,
        )
        agent = ReflectiveCriticOnlyLinearSemanticsAgent(
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
            env_desc_file=env_desc_file,
            jinja2_env=jinja2_env,
            critic_llm_template_name=critic_llm_template_name,
            critic_llm_env_desc_file=critic_llm_env_desc_file,
        )

    agent.max_iterations = num_episodes
    agent.traj_history_last_n = traj_history_last_n

    print("critic_only init done")

    if run_mode == "resume":
        start_episode = restore_agent_from_run(
            agent,
            logdir,
            semantic=True,
            is_qtable=(task == "dist_state_critic_only"),
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
    overall_log_mode = (
        "a" if run_mode == "resume" and os.path.exists(overall_log_path) else "w"
    )
    overall_log_file = open(overall_log_path, overall_log_mode)
    if overall_log_mode == "w":
        overall_log_file.write(
            "Iteration, CPU Time, API Time, Total Episodes, Total Steps, Total Reward\n"
        )
        overall_log_file.flush()

    if start_episode >= num_episodes:
        print(f"Run already has {start_episode} episodes; nothing to resume.")
        overall_log_file.close()
        return

    for episode in range(start_episode, num_episodes):
        print(f"Episode: {episode}")
        curr_episode_dir = f"{logdir}/episode_{episode}"
        os.makedirs(curr_episode_dir, exist_ok=True)
        succeeded = False
        for trial_idx in range(5):
            try:
                cpu_time, api_time, total_episodes, total_steps, total_reward = (
                    agent.train_policy(world, curr_episode_dir)
                )
                overall_log_file.write(
                    f"{episode + 1}, {cpu_time}, {api_time}, {total_episodes}, {total_steps}, {total_reward}\n"
                )
                overall_log_file.flush()
                print(f"{trial_idx + 1}th trial attempt succeeded in training")
                succeeded = True
                break
            except Exception as e:
                print(
                    f"{trial_idx + 1}th trial attempt failed with error in training: {e}"
                )
                traceback.print_exc()
                continue
        if not succeeded:
            print(f"Episode {episode} failed to train after 5 attempts")
            break
    overall_log_file.close()
