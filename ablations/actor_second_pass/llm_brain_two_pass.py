import time as _time

from agent.policy.llm_brain_linear_policy import LLMBrain


class TwoPassLLMBrain(LLMBrain):
    """LLM brain for budget-matched two-pass optimizer ablations."""

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

    def llm_propose_standard(
        self,
        episode_reward_buffer_string,
        parse_parameters,
        step_number,
        rank,
        optimum,
        search_step_size=None,
        actions=None,
    ):
        """Run the original ProPS optimizer prompt."""
        return self.llm_update_parameters_num_optim(
            episode_reward_buffer_string,
            parse_parameters,
            step_number,
            rank=rank,
            optimum=optimum,
            search_step_size=search_step_size,
            actions=actions,
        )
