import gymnasium as gym
import random
import numpy as np
import os
import time
import re
import json
import urllib.request
from jinja2 import Template
from openai import OpenAI
import google.generativeai as genai
import anthropic
import time


class LLMBrain:
    def __init__(
        self,
        llm_si_template: Template,
        llm_output_conversion_template: Template,
        llm_model_name: str,
    ):
        self.llm_si_template = llm_si_template
        self.llm_output_conversion_template = llm_output_conversion_template
        self.llm_conversation = []
        self.llm_model_name = llm_model_name
        model_name_lower = llm_model_name.lower()
        if "gemini" in model_name_lower:
            self.model_group = "gemini"
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        elif model_name_lower.startswith("ollama"):
            self.model_group = "ollama"
            self.ollama_base_url = os.environ.get(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            ).rstrip("/")
            self.ollama_api_key = os.environ.get("OLLAMA_API_KEY")
            if llm_model_name.startswith("ollama/"):
                self.ollama_model_name = llm_model_name.split("ollama/", 1)[1]
            elif llm_model_name.startswith("ollama:"):
                self.ollama_model_name = llm_model_name.split("ollama:", 1)[1]
            else:
                self.ollama_model_name = os.environ.get("OLLAMA_MODEL")
            if not self.ollama_model_name:
                raise ValueError(
                    "For Ollama, use llm_model_name='ollama/<model>' "
                    "or set OLLAMA_MODEL."
                )
        elif model_name_lower.startswith("claude"):
            self.model_group = "anthropic"
            self.client = anthropic.Client(api_key=os.environ["ANTHROPIC_API_KEY"])
        else:
            self.model_group = "openai"
            self.client = OpenAI()

    def _is_non_retryable_model_error(self, error):
        error_msg = str(error).lower()
        non_retryable_fragments = [
            "404",
            "not found",
            "unsupported for generatecontent",
            "unknown model",
            "model_not_found",
            "invalid model",
            "does not exist",
        ]
        return any(fragment in error_msg for fragment in non_retryable_fragments)

    def _is_rate_limit_error(self, error):
        error_msg = str(error).lower()
        rate_limit_fragments = [
            "quota",
            "rate limit",
            "429",
            "resource_exhausted",
            "retry_delay",
            "too many requests",
        ]
        return any(fragment in error_msg for fragment in rate_limit_fragments)

    def _extract_retry_delay_seconds(self, error, default_seconds=60):
        error_text = str(error)
        match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", error_text, re.DOTALL)
        if match:
            return max(1, int(match.group(1)))
        return default_seconds

    def reset_llm_conversation(self):
        self.llm_conversation = []

    def add_llm_conversation(self, text, role):
        if self.model_group in ["openai", "ollama"]:
            self.llm_conversation.append({"role": role, "content": text})
        elif self.model_group == "anthropic":
            self.llm_conversation.append({"role": role, "content": text})
        else:
            self.llm_conversation.append({"role": role, "parts": text})

    def _query_ollama(self, temperature=0):
        # print("OLLAMA URL:", self.ollama_base_url)
        # print("OLLAMA MODEL:", self.ollama_model_name)
        # print("NUM MESSAGES:", len(self.llm_conversation))
        # print("PROMPT CHARS:", sum(len(m.get("content", "")) for m in self.llm_conversation))
        payload = {
            "model": self.ollama_model_name,
            "messages": self.llm_conversation,
            "stream": False,
        }
        payload["options"] = {"temperature": temperature}

        headers = {"Content-Type": "application/json"}
        if self.ollama_api_key:
            headers["Authorization"] = f"Bearer {self.ollama_api_key}"

        request = urllib.request.Request(
            f"{self.ollama_base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        if "error" in response_payload:
            raise RuntimeError(response_payload["error"])
        return response_payload["message"]["content"]

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
                    print(
                        f"Waiting for {wait_seconds} seconds before retrying..."
                    )
                    time.sleep(wait_seconds)

            if self.model_group in ["openai", "ollama"]:
                # add the response to self.llm_conversation
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
                    print(
                        f"Waiting for {wait_seconds} seconds before retrying..."
                    )
                    time.sleep(wait_seconds)

            return responses

    def parse_parameters(self, parameters_string):
        new_parameters_list = []

        # Update the Q-table based on the new Q-table
        for row in parameters_string.split("\n"):
            if row.strip().strip(","):
                try:
                    parameters_row = [
                        float(x.strip().strip(",")) for x in row.split(",")
                    ]
                    new_parameters_list.append(parameters_row)
                except Exception as e:
                    print(e)

        return new_parameters_list

    def llm_update_parameters(self, parameters, replay_buffer, parse_parameters=None):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {
                "replay_buffer_string": str(replay_buffer),
                "parameters_string": str(parameters),
            }
        )

        self.add_llm_conversation(system_prompt, "user")
        new_parameters_with_reasoning = self.query_llm()

        if self.model_group in ["openai", "ollama"]:
            self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        else:
            self.add_llm_conversation(new_parameters_with_reasoning, "model")
        self.add_llm_conversation(
            self.llm_output_conversion_template.render(),
            "user",
        )
        new_parameters = self.query_llm()

        if parse_parameters is None:
            new_parameters_list = self.parse_parameters(new_parameters)
        else:
            new_parameters_list = parse_parameters(new_parameters)

        return new_parameters_list, [new_parameters_with_reasoning, new_parameters]

    def llm_update_parameters_sas(self, episode_reward_buffer, parse_parameters=None):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {"episode_reward_buffer_string": str(episode_reward_buffer)}
        )

        self.add_llm_conversation(system_prompt, "user")
        new_parameters_with_reasoning = self.query_llm()

        print(system_prompt)

        self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        self.add_llm_conversation(
            self.llm_output_conversion_template.render(),
            "user",
        )
        new_parameters = self.query_llm()

        if parse_parameters is None:
            new_parameters_list = self.parse_parameters(new_parameters)
        else:
            new_parameters_list = parse_parameters(new_parameters)

        return new_parameters_list, [
            "system:\n"
            + system_prompt
            + "\n\n\nLLM:\n"
            + new_parameters_with_reasoning,
            new_parameters,
        ]

    def llm_update_parameters_num_optim(
        self,
        episode_reward_buffer,
        parse_parameters,
        step_number,
        rank=None,
        optimum=None,
        search_step_size=0.1,
        actions=None,
        max_iterations=None,
        param_min=-6.0,
        param_max=6.0,
        max_edit_count=None,
        template_context=None,
    ):
        self.reset_llm_conversation()

        prompt_context = {
            "episode_reward_buffer_string": str(episode_reward_buffer),
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

        api_start_time = time.time()
        new_parameters_with_reasoning = self.query_llm()
        api_time = time.time() - api_start_time

        # print(system_prompt)

        # self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        # self.add_llm_conversation(
        #     self.llm_output_conversion_template.render(),
        #     "user",
        # )
        # new_parameters = self.query_llm()
        new_parameters_list = parse_parameters(new_parameters_with_reasoning)

        return (
            new_parameters_list,
            "system:\n"
            + system_prompt
            + "\n\n\nLLM:\n"
            + new_parameters_with_reasoning,
            api_time,
        )

    def llm_update_parameters_num_optim_q_table(
        self,
        episode_reward_buffer,
        parse_parameters,
        step_number,
        actions,
        num_states,
        optimum,
    ):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {
                "episode_reward_buffer_string": str(episode_reward_buffer),
                "step_number": str(step_number),
                "actions": actions,
                "rank": num_states,
                "optimum": str(optimum),
            }
        )

        self.add_llm_conversation(system_prompt, "user")
        new_parameters_with_reasoning = self.query_llm()

        print(system_prompt)

        # self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        # self.add_llm_conversation(
        #     self.llm_output_conversion_template.render(),
        #     "user",
        # )
        # new_parameters = self.query_llm()
        new_parameters_list = parse_parameters(new_parameters_with_reasoning)

        return (
            new_parameters_list,
            "system:\n"
            + system_prompt
            + "\n\n\nLLM:\n"
            + new_parameters_with_reasoning,
        )

    def llm_update_parameters_num_optim_imitation(
        self,
        demonstrations_str,
        episode_reward_buffer,
        parse_parameters,
        step_number,
        search_std,
    ):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {
                "expert_demonstration_string": demonstrations_str,
                "episode_reward_buffer_string": str(episode_reward_buffer),
                "step_number": str(step_number),
                "search_std": str(search_std),
            }
        )

        self.add_llm_conversation(system_prompt, "user")
        new_parameters_with_reasoning = self.query_llm()

        print(system_prompt)

        # self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        # self.add_llm_conversation(
        #     self.llm_output_conversion_template.render(),
        #     "user",
        # )
        # new_parameters = self.query_llm()
        new_parameters_list = parse_parameters(new_parameters_with_reasoning)

        return (
            new_parameters_list,
            "system:\n"
            + system_prompt
            + "\n\n\nLLM:\n"
            + new_parameters_with_reasoning,
        )

    def llm_propose_parameters_num_optim_based_on_anchor(
        self,
        episode_reward_buffer,
        parse_parameters,
        step_number,
        search_std,
        anchor_parameters,
    ):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {
                "episode_reward_buffer_string": str(episode_reward_buffer),
                "step_number": str(step_number),
                "search_std": str(search_std),
                "anchor_parameters": str(anchor_parameters),
            }
        )

        self.add_llm_conversation(system_prompt, "user")
        new_parameters_with_reasoning = self.query_llm()

        print(system_prompt)

        # self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        # self.add_llm_conversation(
        #     self.llm_output_conversion_template.render(),
        #     "user",
        # )
        # new_parameters = self.query_llm()
        new_parameters_list = parse_parameters(new_parameters_with_reasoning)

        return (
            new_parameters_list,
            "system:\n"
            + system_prompt
            + "\n\n\nLLM:\n"
            + new_parameters_with_reasoning,
        )

    def llm_propose_multiple_parameters_num_optim_based_on_anchor(
        self,
        episode_reward_buffer,
        parse_parameters,
        step_number,
        search_std,
        anchor_parameters,
        num_candidates,
        temperature,
    ):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {
                "episode_reward_buffer_string": str(episode_reward_buffer),
                "step_number": str(step_number),
                "search_std": str(search_std),
                "anchor_parameters": str(anchor_parameters),
            }
        )

        # print(system_prompt)
        self.add_llm_conversation(system_prompt, "user")
        new_parameters_with_reasoning_list = self.query_llm_multiple_response(
            num_candidates, temperature
        )
        # print(new_parameters_with_reasoning_list)

        new_parameters_list = []
        reasonings_list = []
        for new_params in new_parameters_with_reasoning_list:
            new_params_np = parse_parameters(new_params)
            new_parameters_list.append(new_params_np)
            reasonings_list.append(new_params)

        return (
            system_prompt,
            new_parameters_list,
            reasonings_list,
        )

    def llm_propose_parameters_num_optim_based_on_anchor_thread(
        self,
        new_candidates,
        new_idx,
        episode_reward_buffer,
        parse_parameters,
        step_number,
        search_std,
        anchor_parameters,
    ):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {
                "episode_reward_buffer_string": str(episode_reward_buffer),
                "step_number": str(step_number),
                "search_std": str(search_std),
                "anchor_parameters": str(anchor_parameters),
            }
        )

        self.add_llm_conversation(system_prompt, "user")
        new_parameters_with_reasoning = self.query_llm()

        print(system_prompt)

        # self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        # self.add_llm_conversation(
        #     self.llm_output_conversion_template.render(),
        #     "user",
        # )
        # new_parameters = self.query_llm()
        new_parameters_list = parse_parameters(new_parameters_with_reasoning)
        new_candidates[new_idx] = new_parameters_list

        return (
            new_parameters_list,
            "system:\n"
            + system_prompt
            + "\n\n\nLLM:\n"
            + new_parameters_with_reasoning,
        )

    def llm_update_parameters_num_optim_semantics(
        self,
        episode_reward_buffer,
        parse_parameters,
        step_number,
        env_desc_file,
        rank=None,
        optimum=None,
        search_step_size=0.1,
        actions=None,
        template_context=None,
    ):
        self.reset_llm_conversation()

        prompt_context = {
            "episode_reward_buffer_string": str(episode_reward_buffer),
            "env_description": env_desc_file,
            "step_number": str(step_number),
            "rank": rank,
            "optimum": str(optimum),
            "step_size": str(search_step_size),
            "actions": actions,
        }
        if template_context:
            prompt_context.update(template_context)

        system_prompt = self.llm_si_template.render(
            prompt_context
        )


        self.add_llm_conversation(system_prompt, "user")

        api_start_time = time.time()
        new_parameters_with_reasoning = self.query_llm()
        api_time = time.time() - api_start_time

        # print(system_prompt)

        # self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        # self.add_llm_conversation(
        #     self.llm_output_conversion_template.render(),
        #     "user",
        # )
        # new_parameters = self.query_llm()
        new_parameters_list = parse_parameters(new_parameters_with_reasoning)

        return (
            new_parameters_list,
            "system:\n"
            + system_prompt
            + "\n\n\nLLM:\n"
            + new_parameters_with_reasoning,
            api_time,
        )
