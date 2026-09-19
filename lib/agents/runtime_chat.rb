# frozen_string_literal: true

module Agents
  # Own the public Chat boundary so RubyLLM's recursive #complete calls remain bounded.
  class RuntimeChat < RubyLLM::Chat
    def initialize(run_context:, **options)
      @run_context = run_context
      # Transport retries would create requests outside the model-call budget.
      context = RubyLLM.context { |config| config.max_retries = 0 }
      super(**options, context: context)
      before_tool_call do
        run_context.execution_budget.consume!(:tool_calls) unless run_context.handoff_pending?
      end
      after_message { |message| record_response(message) if message.role == :assistant }
    end

    def complete(&)
      @run_context.execution_budget.consume!(:model_calls)
      response = super
      @run_context.execution_budget.check_duration!
      response
    end

    private

    def record_response(message)
      @run_context.usage.add(message)
      @run_context.callback_manager.emit_llm_call_complete(
        @run_context.context.fetch(:current_agent), model.id, message, @run_context
      )
    end
  end
end
