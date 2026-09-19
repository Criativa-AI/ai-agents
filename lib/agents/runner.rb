# frozen_string_literal: true

module Agents
  # The execution engine that orchestrates conversations between users and agents.
  # Runner manages the conversation flow, handles tool execution through RubyLLM,
  # coordinates handoffs between agents, and ensures thread-safe operation.
  #
  # The Runner follows a turn-based execution model where each turn consists of:
  # 1. Sending a message to the LLM with current context
  # 2. Receiving a response that may include tool calls
  # 3. Executing tools and getting results (handled by RubyLLM)
  # 4. Checking for agent handoffs
  # 5. Continuing until no more tools are called
  #
  # ## Thread Safety
  # The Runner ensures thread safety by:
  # - Creating new context wrappers for each execution
  # - Using tool wrappers that pass context through parameters
  # - Never storing execution state in shared variables
  #
  # ## Integration with RubyLLM
  # We leverage RubyLLM for LLM communication and tool execution while
  # maintaining our own context management and handoff logic.
  #
  # @example Simple conversation
  #   agent = Agents::Agent.new(
  #     name: "Assistant",
  #     instructions: "You are a helpful assistant",
  #     tools: [weather_tool]
  #   )
  #
  #   result = Agents::Runner.run(agent, "What's the weather?")
  #   puts result.output
  #   # => "Let me check the weather for you..."
  #
  # @example Conversation with context
  #   result = Agents::Runner.run(
  #     support_agent,
  #     "I need help with my order",
  #     context: { user_id: 123, order_id: 456 }
  #   )
  #
  # @example Multi-agent handoff
  #   triage = Agents::Agent.new(
  #     name: "Triage",
  #     instructions: "Route users to the right specialist",
  #     handoff_agents: [billing_agent, tech_agent]
  #   )
  #
  #   result = Agents::Runner.run(triage, "I can't pay my bill")
  #   # Triage agent will handoff to billing_agent
  class Runner
    DEFAULT_MAX_TURNS = 10

    class MaxTurnsExceeded < StandardError; end
    class AgentNotFoundError < StandardError; end

    # Create a thread-safe agent runner for multi-agent conversations.
    # The first agent becomes the default entry point for new conversations.
    # All agents must be explicitly provided - no automatic discovery.
    #
    # @param agents [Array<Agents::Agent>] All agents that should be available for handoffs
    # @return [AgentRunner] Thread-safe runner that can be reused across multiple conversations
    #
    # @example
    #   runner = Agents::Runner.with_agents(triage_agent, billing_agent, support_agent)
    #   result = runner.run("I need help")  # Uses triage_agent for new conversation
    #   result = runner.run("More help", context: stored_context)  # Continues with appropriate agent
    def self.with_agents(*agents)
      AgentRunner.new(agents)
    end

    # Execute an agent with the given input and context.
    # This is now called internally by AgentRunner and should not be used directly.
    #
    # @param starting_agent [Agents::Agent] The agent to run
    # @param input [String] The user's input message
    # @param context [Hash] Shared context data accessible to all tools
    # @param registry [Hash] Registry of agents for handoff resolution
    # @param max_turns [Integer] Maximum conversation turns before stopping
    # @param headers [Hash, nil] Custom HTTP headers passed to the underlying LLM provider
    # @param params [Hash, nil] Provider-specific parameters passed to the underlying LLM (e.g., service_tier)
    # @param callbacks [Hash] Optional callbacks for real-time event notifications
    # @return [RunResult] The result containing output, messages, and usage
    ExecutionState = Struct.new(:agent, :input, :context, :options, :chat, :turn, keyword_init: true)
    RUN_DEFAULTS = { context: {}, registry: {}, max_turns: DEFAULT_MAX_TURNS, headers: nil, params: nil,
                     callbacks: {}, limits: {}, execution_budget: nil }.freeze

    def run(starting_agent, input, **options)
      state = build_execution_state(starting_agent, input, options)
      state.context.callback_manager.emit_run_start(starting_agent.name, input, state.context)
      prepare_chat(state)
      execute_turns(state)
    rescue MaxTurnsExceeded => e
      finalize_execution(state, output: "Conversation ended: #{e.message}", error: e)
    rescue StandardError => e
      raise unless state

      finalize_execution(state, output: nil, error: e)
    end

    private

    def build_execution_state(agent, input, options)
      unknown = options.keys - RUN_DEFAULTS.keys
      raise ArgumentError, "Unknown run options: #{unknown.join(", ")}" unless unknown.empty?

      options = RUN_DEFAULTS.merge(options)
      context = build_run_context(options)
      context.context[:current_agent] = agent.name
      ExecutionState.new(agent: agent, input: input, context: context, options: options, turn: 0)
    end

    def build_run_context(options)
      budget = ExecutionBudget.new(**options[:limits], max_model_calls: options[:max_turns],
                                                       parent: options[:execution_budget])
      RunContext.new(deep_copy_context(options[:context]), callbacks: options[:callbacks], execution_budget: budget)
    end

    def prepare_chat(state)
      agent = state.agent
      state.chat = RuntimeChat.new(run_context: state.context, model: agent.model,
                                   provider: agent.provider, assume_model_exists: agent.assume_model_exists)
      apply_runtime_options(state)
      configure_chat_for_agent(state.chat, state.agent, state.context, replace: false)
      restore_conversation_history(state.chat, state.context)
      emit_chat_created(state)
    end

    def execute_turns(state)
      loop do
        response = execute_turn(state)
        if response.is_a?(RubyLLM::Tool::Halt) && state.context.handoff_pending?
          switch_agent(state)
          next
        end
        next if !response.is_a?(RubyLLM::Tool::Halt) && response.tool_call?

        return finalize_execution(state, output: response.content)
      end
    end

    def execute_turn(state)
      state.turn += 1
      max_turns = state.options[:max_turns]
      raise MaxTurnsExceeded, "Exceeded maximum turns: #{max_turns}" if state.turn > max_turns

      count = chat_message_count(state.chat)
      emit_agent_thinking(state)
      response = request_response(state)
      assign_agent_name_to_new_assistant_messages(state.chat, state.agent, count)
      response
    end

    def emit_agent_thinking(state)
      input = state.turn == 1 ? state.input : "(continuing conversation)"
      state.context.callback_manager.emit_agent_thinking(state.agent.name, input, state.context)
    end

    def request_response(state)
      if state.turn == 1 && !last_message_matches?(state.chat, state.input)
        state.chat.ask(state.input)
      else
        state.chat.complete
      end
    end

    def switch_agent(state)
      handoff = state.context.take_pending_handoff
      target = handoff[:target_agent]
      unless state.options[:registry][target.name]
        raise AgentNotFoundError, "Handoff failed: Agent '#{target.name}' not found in registry"
      end

      complete_handoff(state, target, handoff)
      state.agent = target
      configure_handoff_chat(state)
    end

    def configure_handoff_chat(state)
      state.context.context[:current_agent] = state.agent.name
      configure_chat_for_agent(state.chat, state.agent, state.context, replace: true)
      apply_runtime_options(state)
      emit_chat_created(state)
    end

    def complete_handoff(state, target, handoff)
      agent = state.agent
      context = state.context
      handoff_relationship(agent, target)&.call_hook(context, handoff)
      save_conversation_state(state.chat, context, agent)
      callbacks = context.callback_manager
      callbacks.emit_agent_complete(agent.name, nil, nil, context)
      callbacks.emit_agent_handoff(agent.name, target.name, handoff[:reason] || "handoff", context, handoff[:metadata])
    end

    def apply_runtime_options(state)
      headers = merged_option(state, :headers)
      params = merged_option(state, :params)
      apply_headers(state.chat, headers)
      apply_params(state.chat, params)
    end

    def merged_option(state, key)
      agent_value = Helpers::HashNormalizer.normalize(state.agent.public_send(key), label: key.to_s)
      runtime_value = Helpers::HashNormalizer.normalize(state.options[key], label: key.to_s)
      Helpers::HashNormalizer.merge(agent_value, runtime_value)
    end

    def emit_chat_created(state)
      state.context.callback_manager.emit_chat_created(
        state.chat, state.agent.name, state.agent.model, state.context, state.agent.temperature
      )
    end

    def finalize_execution(state, **result)
      finalize_run(state.chat, state.context, state.agent, **result)
    end

    # Saves conversation state, builds a RunResult, emits completion callbacks, and returns it.
    # Centralises the finalize-and-return pattern used by the normal path, halt path, and error rescues.
    #
    # @param chat [RubyLLM::Chat, nil] The chat instance (nil in early-failure rescues)
    # @param context_wrapper [RunContext] Context wrapper for state and callbacks
    # @param current_agent [Agents::Agent] The currently active agent
    # @param output [String, nil] The output text for the result
    # @param error [StandardError, nil] Optional error to attach to the result
    # @return [RunResult]
    def finalize_run(chat, context_wrapper, current_agent, output:, error: nil)
      save_conversation_state(chat, context_wrapper, current_agent) if chat
      context_wrapper.context[:execution_counts] = context_wrapper.execution_budget.snapshot

      result = RunResult.new(
        output: output,
        messages: chat ? Helpers::MessageExtractor.extract_messages(chat, current_agent) : [],
        usage: context_wrapper.usage,
        error: error,
        context: context_wrapper.context
      )

      context_wrapper.callback_manager.emit_agent_complete(current_agent.name, result, error, context_wrapper)
      context_wrapper.callback_manager.emit_run_complete(current_agent.name, result, context_wrapper)

      result
    end

    # Creates a deep copy of context data for thread safety.
    # Preserves conversation history array structure while avoiding agent mutation.
    #
    # @param context [Hash] The context to copy
    # @return [Hash] Thread-safe deep copy of the context
    def deep_copy_context(context)
      # Handle deep copying for thread safety
      context.dup.tap do |copied|
        copied[:conversation_history] = context[:conversation_history]&.map(&:dup) || []
        # Don't copy agents - they're immutable
        copied[:current_agent] = context[:current_agent]
        copied[:turn_count] = context[:turn_count] || 0
      end
    end

    # Restores conversation history from context into RubyLLM chat.
    # Converts stored message hashes back into RubyLLM::Message objects with proper content handling.
    # Supports user, assistant, and tool role messages for complete conversation continuity.
    #
    # @param chat [RubyLLM::Chat] The chat instance to restore history into
    # @param context_wrapper [RunContext] Context containing conversation history
    def restore_conversation_history(chat, context_wrapper)
      history = context_wrapper.context[:conversation_history] || []
      valid_tool_call_ids = Set.new

      history.each do |msg|
        next unless restorable_message?(msg)

        next if orphan_tool_message?(msg, valid_tool_call_ids)

        message_params = build_message_params(msg)
        next unless message_params # Skip invalid messages

        message = RubyLLM::Message.new(**message_params)
        assign_restored_agent_name(message, msg)
        chat.add_message(message)

        valid_tool_call_ids.merge(message_params.fetch(:tool_calls, {}).keys) if message.role == :assistant
      end
    end

    def orphan_tool_message?(msg, valid_tool_call_ids)
      return false unless msg[:role].to_sym == :tool && msg[:tool_call_id]
      return false if valid_tool_call_ids.include?(msg[:tool_call_id])

      Agents.logger&.warn("Skipping tool message without matching assistant tool_call_id #{msg[:tool_call_id]}")
      true
    end

    # Check if a message should be restored
    def restorable_message?(msg)
      role = msg[:role].to_sym
      return false unless %i[user assistant tool].include?(role)

      # Allow assistant messages that only contain tool calls (no text content)
      tool_calls_present = role == :assistant && msg[:tool_calls] && !msg[:tool_calls].empty?
      return false if role != :tool && !tool_calls_present &&
                      Helpers::MessageExtractor.content_empty?(msg[:content])

      true
    end

    # Build message parameters for restoration
    def build_message_params(msg)
      role = msg[:role].to_sym
      return nil if role == :tool && !valid_tool_message?(msg)

      params = { role: role, content: build_content(restored_content(msg)) }
      params[:tool_call_id] = msg[:tool_call_id] if role == :tool
      params[:tool_calls] = restored_tool_calls(msg[:tool_calls]) if assistant_tool_calls?(msg)
      params
    end

    def assistant_tool_calls?(msg)
      msg[:role].to_sym == :assistant && msg[:tool_calls] && !msg[:tool_calls].empty?
    end

    def restored_content(msg)
      # Assistant tool-call messages need placeholder content when text is absent.
      return "" if msg[:content].nil? && assistant_tool_calls?(msg)

      msg[:content]
    end

    def restored_tool_calls(tool_calls)
      tool_calls.each_with_object({}) do |tool_call, result|
        id = tool_call[:id] || tool_call["id"]
        next unless id

        result[id] = RubyLLM::ToolCall.new(id: id, name: tool_call[:name] || tool_call["name"],
                                           arguments: tool_call[:arguments] || tool_call["arguments"] || {})
      end
    end

    # Normalize stored content for RubyLLM, preserving prebuilt content and handling multimodal arrays.
    # Multimodal arrays follow the OpenAI content format: [{type: 'text', text: '...'}, {type: 'image_url', ...}]
    def build_content(content_value)
      return content_value if content_value.is_a?(RubyLLM::Content)
      return RubyLLM::Content.new(content_value) unless content_value.is_a?(Array)

      text_parts = text_parts(content_value)
      image_urls = image_urls(content_value)

      return RubyLLM::Content.new(content_value.to_json) if text_parts.empty? && image_urls.empty?

      text = text_parts.join(" ")
      image_urls.any? ? RubyLLM::Content.new(text, image_urls) : RubyLLM::Content.new(text)
    end

    def text_parts(content)
      content_parts(content, "text").filter_map { |part| part[:text] || part["text"] }
    end

    def image_urls(content)
      content_parts(content, "image_url").filter_map do |part|
        part.dig(:image_url, :url) || part.dig("image_url", "url")
      end
    end

    def content_parts(content, type)
      content.select { |part| (part[:type] || part["type"]) == type }
    end

    # Validate tool message has required tool_call_id
    def valid_tool_message?(msg)
      if msg[:tool_call_id]
        true
      else
        Agents.logger&.warn("Skipping tool message without tool_call_id in conversation history")
        false
      end
    end

    # Saves current conversation state from RubyLLM chat back to context for persistence.
    # Maintains conversation continuity across agent handoffs and process boundaries.
    #
    # @param chat [RubyLLM::Chat] The chat instance to extract state from
    # @param context_wrapper [RunContext] Context to save state into
    # @param current_agent [Agents::Agent] The currently active agent
    def save_conversation_state(chat, context_wrapper, current_agent)
      # Extract messages from chat
      messages = Helpers::MessageExtractor.extract_messages(chat, current_agent)

      # Update context with latest state
      context_wrapper.context[:conversation_history] = messages
      context_wrapper.context[:current_agent] = current_agent.name
      context_wrapper.context[:turn_count] = (context_wrapper.context[:turn_count] || 0) + 1
      context_wrapper.context[:last_updated] = Time.now.getlocal

      # Clean up temporary handoff state
      context_wrapper.clear_pending_handoff
    end

    def assign_agent_name_to_new_assistant_messages(chat, current_agent, start_index)
      # Runtime chats are RubyLLM::Chat instances and expose messages. Keep this
      # no-op guard for chat-like doubles/adapters that do not expose history.
      return unless chat.respond_to?(:messages)

      chat.messages[start_index..]&.each do |message|
        next unless message.role == :assistant

        Helpers::MessageExtractor.assign_agent_name(message, current_agent.name)
      end
    end

    def chat_message_count(chat)
      # Runtime chats are RubyLLM::Chat instances and expose messages. Keep this
      # fallback for chat-like doubles/adapters where attribution is irrelevant.
      return 0 unless chat.respond_to?(:messages)

      chat.messages.length
    end

    def assign_restored_agent_name(message, msg)
      return unless message.role == :assistant

      restored_agent_name = msg[:agent_name] || msg["agent_name"]
      Helpers::MessageExtractor.assign_agent_name(message, restored_agent_name)
    end

    # Configures a RubyLLM chat instance with agent-specific settings.
    # Uses RubyLLM's replace option to swap agent context while preserving conversation history during handoffs.
    #
    # @param chat [RubyLLM::Chat] The chat instance to configure
    # @param agent [Agents::Agent] The agent whose configuration to apply
    # @param context_wrapper [RunContext] Thread-safe context wrapper
    # @param replace [Boolean] Whether to replace existing configuration (true for handoffs, false for initial setup)
    # @return [RubyLLM::Chat] The configured chat instance
    def configure_chat_for_agent(chat, agent, context_wrapper, replace: false)
      # Get system prompt (may be dynamic)
      system_prompt = agent.get_system_prompt(context_wrapper)

      # Combine all tools - both handoff and regular tools need wrapping
      all_tools = build_agent_tools(agent, context_wrapper)

      # Switch model if different (important for handoffs between agents using different models)
      if replace
        chat.with_model(
          agent.model,
          provider: agent.provider,
          assume_exists: agent.assume_model_exists
        )
      end

      # Configure chat with instructions, temperature, tools, and schema
      chat.with_instructions(system_prompt, replace: replace) if system_prompt
      chat.with_temperature(agent.temperature) if agent.temperature
      chat.with_tools(*all_tools, replace: replace)
      chat.with_schema(agent.response_schema) if agent.response_schema

      chat
    end

    # Check if the last message in the chat already matches the user's input.
    # This happens when an external system (e.g. Chatwoot) includes the current
    # user message in the conversation history passed via context.
    #
    # TODO: This .to_s == .to_s comparison is a best-effort safety net and is
    # brittle for edge cases (trailing whitespace, Hash/JSON round-tripping).
    # The proper fix is for callers to pass nil when input is already present
    # in conversation history, similar to the handoff continuation path.
    def last_message_matches?(chat, input)
      return false unless input && chat.respond_to?(:messages)

      last_msg = chat.messages.last
      last_msg && last_msg.role == :user && last_msg.content.to_s == input.to_s
    end

    def apply_headers(chat, headers)
      return if headers.empty?

      chat.with_headers(**headers)
    end

    def apply_params(chat, params)
      return if params.empty?

      chat.with_params(**params)
    end

    # Builds thread-safe tool wrappers for an agent's tools and handoff tools.
    #
    # @param agent [Agents::Agent] The agent whose tools to wrap
    # @param context_wrapper [RunContext] Thread-safe context wrapper for tool execution
    # @return [Array<ToolWrapper>] Array of wrapped tools ready for RubyLLM
    def build_agent_tools(agent, context_wrapper)
      handoff_tools = if agent.is_a?(Agent)
                        agent.handoffs.map { |handoff| handoff.build_tool(source_agent: agent) }
                      else
                        agent.handoff_agents.map { |target_agent| HandoffTool.new(target_agent) }
                      end
      all_tools = handoff_tools.map { |tool| ToolWrapper.new(tool, context_wrapper) }

      # Add regular tools
      agent.tools.each do |tool|
        all_tools << ToolWrapper.new(tool, context_wrapper)
      end

      all_tools
    end

    def handoff_relationship(agent, target_agent)
      return unless agent.is_a?(Agent)

      agent.handoff_for(target_agent)
    end
  end
end
