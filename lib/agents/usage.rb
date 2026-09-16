# frozen_string_literal: true

module Agents
  # Usage tracks token consumption across all LLM calls within a single run.
  # This is very rudimentary usage reporting.
  # We can use this further for billing purposes, but is not a replacement for tracing.
  #
  # @example Accumulating usage from multiple LLM calls
  #   usage = Agents::RunContext::Usage.new
  #
  #   # Add usage from first call
  #   usage.add(OpenStruct.new(input_tokens: 100, output_tokens: 50, total_tokens: 150))
  #
  #   # Add usage from second call
  #   usage.add(OpenStruct.new(input_tokens: 200, output_tokens: 100, total_tokens: 300))
  #
  #   puts usage.total_tokens  # => 450
  class Usage
    attr_accessor :input_tokens, :output_tokens, :total_tokens

    # Initialize a new Usage tracker with all counters at zero
    def initialize
      @input_tokens = 0
      @output_tokens = 0
      @total_tokens = 0
    end

    # Add usage metrics from an LLM response to the running totals.
    # Only tracks usage for responses that have token data (e.g., RubyLLM::Message).
    # Safely skips responses without token methods (e.g., RubyLLM::Tool::Halt).
    #
    # @param response [RubyLLM::Message] A RubyLLM::Message object with token usage data
    # @example Adding usage from an LLM response
    #   usage.add(llm_response)
    def add(response)
      return unless response.respond_to?(:input_tokens)

      input = response.input_tokens || 0
      output = response.output_tokens || 0

      @input_tokens += input
      @output_tokens += output
      @total_tokens += input + output
    end
  end
end
