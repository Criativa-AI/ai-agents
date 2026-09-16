# frozen_string_literal: true

# RunContext encapsulates the execution context and usage metrics for a single agent run.
# It provides isolation between concurrent executions by giving each run its own context
# copy and tracking token usage throughout the execution. This is a key component in
# ensuring thread safety.
#
# @example Creating a RunContext for an agent execution
#   context_data = { user_id: 123, session: "abc" }
#   run_context = Agents::RunContext.new(context_data)
#
#   # Access context during execution
#   user_id = run_context.context[:user_id]
#
#   # Track usage after LLM calls
#   run_context.usage.add(llm_response.usage)
#
# @example Tracking token usage across multiple LLM calls
#   run_context = Agents::RunContext.new({})
#
#   # First LLM call
#   response1 = llm.complete(prompt1)
#   run_context.usage.add(response1.usage)
#
#   # Second LLM call
#   response2 = llm.complete(prompt2)
#   run_context.usage.add(response2.usage)
#
#   # Total usage is automatically accumulated
#   puts "Total tokens: #{run_context.usage.total_tokens}"
#
# @example Thread safety through context isolation
#   # Shared configuration (never modified)
#   base_context = { api_key: "secret", model: "gpt-4" }
#
#   # Concurrent agent runs using Async
#   Async do
#     5.times.map do |i|
#       Async do
#         # Each run gets its own context COPY
#         run_context = Agents::RunContext.new(base_context.dup)
#
#         # Safe to modify - changes are isolated to this run
#         run_context.context[:user_id] = i
#         run_context.context[:session] = "session_#{i}"
#
#         # Other concurrent runs cannot see these changes
#         puts "Run #{i}: user_id = #{run_context.context[:user_id]}"
#       end
#     end.map(&:wait)
#   end
#
#   # Key points:
#   # - base_context remains unmodified
#   # - Each run has isolated context via .dup
#   # - No race conditions or data leakage between runs
require_relative "usage"

module Agents
  # Per-run state, handoff coordination, and accumulated usage.
  class RunContext
    Usage = Agents::Usage

    attr_reader :context, :usage, :callbacks, :callback_manager, :execution_budget

    # Initialize a new RunContext with execution context and usage tracking
    #
    # @param context [Hash] The execution context data (will be duplicated for isolation)
    # @param callbacks [Hash] Optional callbacks for real-time event notifications
    def initialize(context, callbacks: {}, execution_budget: ExecutionBudget.new)
      @context = context
      @usage = Usage.new
      @execution_budget = execution_budget
      @callbacks = callbacks || {}
      @callback_manager = CallbackManager.new(@callbacks)
      @handoff_mutex = Mutex.new
    end

    # Store the first handoff accepted during the current execution step.
    # Concurrent calls are serialized so later handoffs cannot overwrite it.
    #
    # @param handoff_info [Hash] Target and optional handoff data
    # @return [Boolean] true when accepted, false when another handoff is pending
    def prepare_handoff(handoff_info)
      @handoff_mutex.synchronize do
        return false if @context[:pending_handoff]

        @execution_budget.consume!(:handoffs)
        @context[:pending_handoff] = handoff_info
        true
      end
    end

    def handoff_pending?
      @handoff_mutex.synchronize { !!@context[:pending_handoff] }
    end

    # Atomically remove and return the pending handoff.
    #
    # @return [Hash, nil] The pending handoff information
    def take_pending_handoff
      @handoff_mutex.synchronize { @context.delete(:pending_handoff) }
    end

    # Clear temporary handoff state after a run is finalized.
    def clear_pending_handoff
      @handoff_mutex.synchronize { @context.delete(:pending_handoff) }
    end
  end
end
