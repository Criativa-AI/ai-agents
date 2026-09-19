# frozen_string_literal: true

module Agents
  # Admission limits for one run, including recursive model calls and nested agents.
  # Deadlines stop new work; they do not interrupt an external request already in flight.
  class ExecutionBudget
    class Exceeded < Error; end

    def initialize(max_model_calls: nil, max_tool_calls: nil, max_handoffs: nil, max_duration: nil, parent: nil)
      @limits = { model_calls: max_model_calls, tool_calls: max_tool_calls, handoffs: max_handoffs }
      validate_count_limits!
      validate_duration!(max_duration)
      @started_at = monotonic_time
      @deadline = @started_at + max_duration if max_duration
      @counts = { model_calls: 0, tool_calls: 0, handoffs: 0 }
      @parent = parent
      @mutex = Mutex.new
    end

    def consume!(kind)
      @mutex.synchronize do
        check_duration!
        limit = @limits.fetch(kind)
        exceeded!(kind, limit) if limit && @counts.fetch(kind) >= limit
        @parent&.consume!(kind)
        @counts[kind] += 1
      end
    end

    def check_duration!
      @parent&.check_duration!
      raise Exceeded, "Execution duration exceeded" if @deadline && monotonic_time >= @deadline
    end

    def snapshot
      @mutex.synchronize { @counts.merge(elapsed_seconds: monotonic_time - @started_at) }
    end

    private

    def validate_count_limits!
      @limits.each_value do |limit|
        next if limit.nil? || (limit.is_a?(Integer) && limit >= 0)

        raise ArgumentError, "Execution limits must be non-negative integers"
      end
    end

    def validate_duration!(duration)
      return if duration.nil? || (duration.is_a?(Numeric) && duration.finite? && duration >= 0)

      raise ArgumentError, "max_duration must be a finite, non-negative number"
    end

    def monotonic_time
      Process.clock_gettime(Process::CLOCK_MONOTONIC)
    end

    def exceeded!(kind, limit)
      raise Runner::MaxTurnsExceeded, "Exceeded maximum turns: #{limit}" if kind == :model_calls

      raise Exceeded, "Exceeded maximum #{kind}: #{limit}"
    end
  end
end
