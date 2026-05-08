import time as _time

from agent.policy.llm_brain_linear_policy import LLMBrain


class ReflectiveLLMBrain(LLMBrain):
    """LLM brain with two-phase propose-then-reflect pipeline.

    Phase 1: Search-LLM proposes parameters (ProPS-style, using existing templates).
    Phase 2: After environment evaluation, Critic-LLM sees the outcome and proposes a revision.
    """

    def query_llm(self):
        for attempt in range(10):
            try:
                if self.model_group == "openai":
                    completion = self.client.chat.completions.create(
                        model=self.llm_model_name,
                        messages=self.llm_conversation,
                    )
                    response = completion.choices[0].message.content
                elif self.model_group == "ollama":
                    response = self._query_ollama()
                elif self.model_group == "anthropic":
                    message = self.client.messages.create(
                        model=self.llm_model_name,
                        messages=self.llm_conversation,
                        max_tokens=1024,
                    )
                    response = message.content[0].text
                else:
                    import google.generativeai as genai
                    model = genai.GenerativeModel(model_name=self.llm_model_name)
                    chat_session = model.start_chat(history=self.llm_conversation[:-1])
                    response = chat_session.send_message(
                        self.llm_conversation[-1]["parts"]
                    )
                    response = response.text
            except Exception as e:
                print(f"Error: {e}")
                if self._is_non_retryable_model_error(e):
                    raise
                print("Retrying...")
                if attempt == 9:
                    raise Exception("Failed")
                else:
                    wait_seconds = (
                        self._extract_retry_delay_seconds(e)
                        if self._is_rate_limit_error(e)
                        else 60
                    )
                    print(f"Waiting for {wait_seconds} seconds before retrying...")
                    _time.sleep(wait_seconds)
                    continue

            if self.model_group in ["openai", "ollama"]:
                self.add_llm_conversation(response, "assistant")
            else:
                self.add_llm_conversation(response, "model")

            return response

    def query_llm_multiple_response(self, num_responses, temperature):
        for attempt in range(5):
            try:
                if self.model_group == "openai":
                    completion = self.client.chat.completions.create(
                        model=self.llm_model_name,
                        messages=self.llm_conversation,
                        n=num_responses,
                        temperature=temperature,
                    )
                    responses = [
                        completion.choices[i].message.content
                        for i in range(num_responses)
                    ]
                elif self.model_group == "ollama":
                    responses = [
                        self._query_ollama(temperature=temperature)
                        for _ in range(num_responses)
                    ]
                else:
                    import google.generativeai as genai
                    model = genai.GenerativeModel(model_name=self.llm_model_name)
                    responses = model.generate_content(
                        contents=self.llm_conversation,
                        generation_config=genai.GenerationConfig(
                            candidate_count=num_responses,
                            temperature=temperature,
                        ),
                    )
                    responses = [
                        "\n".join([x.text for x in c.content.parts])
                        for c in responses.candidates
                    ]
            except Exception as e:
                print(f"Error: {e}")
                if self._is_non_retryable_model_error(e):
                    raise
                print("Retrying...")
                if attempt == 4:
                    raise Exception("Failed")
                else:
                    wait_seconds = (
                        self._extract_retry_delay_seconds(e)
                        if self._is_rate_limit_error(e)
                        else 60
                    )
                    print(f"Waiting for {wait_seconds} seconds before retrying...")
                    _time.sleep(wait_seconds)
                    continue

            return responses

    def search_llm_propose_initial(
        self,
        episode_reward_buffer_string,
        parse_parameters,
        step_number,
        rank,
        optimum,
        search_step_size=None,
        actions=None,
        max_iterations=None,
        param_min=-6.0,
        param_max=6.0,
        max_edit_count=None,
        template_context=None,
    ):
        """Phase 1: Call Search-LLM to propose initial parameters."""
        self.reset_llm_conversation()

        prompt_context = {
            "episode_reward_buffer_string": str(episode_reward_buffer_string),
            "step_number": str(step_number),
            "rank": rank,
            "optimum": str(optimum),
            "step_size": str(search_step_size),
            "actions": actions,
            "max_iterations": (
                "" if max_iterations is None else str(max_iterations)
            ),
            "param_min": str(param_min),
            "param_max": str(param_max),
            "max_edit_count": max_edit_count,
        }
        if template_context:
            prompt_context.update(template_context)

        system_prompt = self.llm_si_template.render(prompt_context)

        self.add_llm_conversation(system_prompt, "user")
        api_start_time = _time.time()
        new_parameters_with_reasoning = self.query_llm()
        api_time = _time.time() - api_start_time

        new_parameters_list = parse_parameters(new_parameters_with_reasoning)

        return (
            new_parameters_list,
            "system:\n"
            + system_prompt
            + "\n\n\nLLM:\n"
            + new_parameters_with_reasoning,
            api_time,
        )

    def critic_llm_reflect_and_revise(
        self,
        parse_parameters,
        jinja2_env,
        critic_llm_template_name,
        critic_llm_context,
    ):
        """Phase 2: Call Critic-LLM to reflect on the outcome and propose revised parameters.

        Args:
            parse_parameters: function to extract params from LLM response text
            jinja2_env: Jinja2 Environment for template loading
            critic_llm_template_name: path to the reflection template
            critic_llm_context: dict with proposed_params, achieved_reward,
                          trajectory_summary, stats, history, env_description

        Returns (revised_params, reasoning, api_time).
        """
        critic_llm_template = jinja2_env.get_template(critic_llm_template_name)
        critic_llm_prompt = critic_llm_template.render(critic_llm_context)

        self.reset_llm_conversation()
        self.add_llm_conversation(critic_llm_prompt, "user")
        t0 = _time.time()
        revision_response = self.query_llm()
        api_time = _time.time() - t0

        revised_params = parse_parameters(revision_response)
        reasoning = (
            "Critic-LLM (Reflection) prompt:\n"
            + critic_llm_prompt
            + "\n\nCritic-LLM response:\n"
            + revision_response
        )
        return revised_params, reasoning, api_time
