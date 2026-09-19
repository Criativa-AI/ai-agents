# frozen_string_literal: true

require "spec_helper"
require "webmock/rspec"

RSpec.describe Agents::Runner do
  include OpenAITestHelper

  let(:effects) { [] }
  let(:tool) do
    recorded = effects
    Class.new(Agents::Tool) do
      define_method(:name) { "record_effect" }
      define_method(:description) { "Record an effect" }
      define_method(:perform) do |_context|
        recorded << "completed"
        "recorded"
      end
    end.new
  end
  let(:agent) { Agents::Agent.new(name: "Principal", model: "gpt-4o", tools: [tool]) }
  let(:specialist) { Agents::Agent.new(name: "Specialist", model: "gpt-4o") }
  let(:tool_response) { { tool_calls: [{ name: tool.name, arguments: {} }] } }

  before do
    setup_openai_test_config
    disable_net_connect!
  end

  it "counts recursive model calls against max_turns before the next request" do
    request = stub_chat_sequence(tool_response, tool_response, "Done")

    result = described_class.with_agents(agent).run("Help", max_turns: 1)

    expect(result.error).to be_a(Agents::Runner::MaxTurnsExceeded)
    expect(request).to have_been_requested.once
    expect(effects).to eq(["completed"])
    expect(result.messages).to include(hash_including(role: :tool, content: "recorded"))
  end

  it "counts every model response and emits each completion once" do
    stub_chat_sequence(tool_response, "Done")
    completions = []
    runner = described_class.with_agents(agent)
    runner.on_llm_call_complete { |_name, _model, response| completions << response }

    result = runner.run("Help")

    expect(result).to be_success
    expect(result.usage).to have_attributes(input_tokens: 30, output_tokens: 13, total_tokens: 43)
    expect(completions.size).to eq(2)
  end

  it "does not execute remaining source tools after an accepted handoff" do
    agent.register_handoffs(specialist)
    stub_chat_sequence(
      { tool_calls: [{ name: "handoff_to_specialist", arguments: {} }, { name: tool.name, arguments: {} }] },
      "Specialist done"
    )

    result = described_class.with_agents(agent, specialist).run("Help")

    expect(result).to be_success
    expect(result.output).to eq("Specialist done")
    expect(effects).to be_empty
    expect(result.messages).to include(hash_including(role: :tool, content: include("not executed")))
  end

  it "limits tools within one response and preserves already completed results" do
    request = stub_chat_sequence({ tool_calls: Array.new(3) { { name: tool.name, arguments: {} } } }, "Done")

    result = described_class.with_agents(agent).run("Help", limits: { max_tool_calls: 2 })

    expect(result.error).to be_a(Agents::ExecutionBudget::Exceeded)
    expect(effects).to eq(%w[completed completed])
    expect(request).to have_been_requested.once
    expect(result.messages.count { |message| message[:role] == :tool }).to eq(2)
    expect(result.context[:execution_counts]).to include(model_calls: 1, tool_calls: 2, handoffs: 0)
    expect(result.usage.total_tokens).to eq(25)
  end

  it "checks the handoff budget before accepting or invoking the destination hook" do
    hook = instance_spy(Proc)
    agent.register_handoff(specialist, on_handoff: hook)
    request = stub_chat_sequence({ tool_calls: [{ name: "handoff_to_specialist", arguments: {} }] }, "Done")

    result = described_class.with_agents(agent, specialist).run("Help", limits: { max_handoffs: 0 })

    expect(result.error).to be_a(Agents::ExecutionBudget::Exceeded)
    expect(hook).not_to have_received(:call)
    expect(request).to have_been_requested.once
    expect(result.context[:current_agent]).to eq("Principal")
  end

  it "keeps the same model budget across handoffs" do
    agent.register_handoffs(specialist)
    request = stub_chat_sequence({ tool_calls: [{ name: "handoff_to_specialist", arguments: {} }] }, "Done")

    result = described_class.with_agents(agent, specialist).run("Help", max_turns: 1)

    expect(result.error).to be_a(Agents::Runner::MaxTurnsExceeded)
    expect(request).to have_been_requested.once
    expect(result.usage.total_tokens).to eq(25)
    expect(result.context[:execution_counts]).to include(model_calls: 1, tool_calls: 1, handoffs: 1)
  end

  it "stops before the first request when the deadline has expired" do
    request = stub_simple_chat("Done")

    result = described_class.with_agents(agent).run("Help", limits: { max_duration: 0 })

    expect(result.error).to be_a(Agents::ExecutionBudget::Exceeded)
    expect(request).not_to have_been_requested
    expect(effects).to be_empty
  end

  it "stops the next tool when time expires without undoing completed work" do
    now = 10.0
    allow(Process).to receive(:clock_gettime).and_call_original
    allow(Process).to receive(:clock_gettime).with(Process::CLOCK_MONOTONIC).and_return(now)
    stub_chat_sequence({ tool_calls: Array.new(2) { { name: tool.name, arguments: {} } } }, "Done")
    runner = described_class.with_agents(agent)
    runner.on_tool_complete do
      allow(Process).to receive(:clock_gettime).with(Process::CLOCK_MONOTONIC).and_return(now + 5)
    end

    result = runner.run("Help", limits: { max_duration: 5 })

    expect(result.error).to be_a(Agents::ExecutionBudget::Exceeded)
    expect(effects).to eq(["completed"])
    expect(result.messages).to include(hash_including(role: :tool, content: "recorded"))
  end

  it "accounts for a response arriving after the deadline and reports the timeout" do
    allow(Process).to receive(:clock_gettime).and_call_original
    allow(Process).to receive(:clock_gettime).with(Process::CLOCK_MONOTONIC).and_return(10.0)
    request = stub_simple_chat("Late answer")
    runner = described_class.with_agents(agent)
    runner.on_llm_call_complete do
      allow(Process).to receive(:clock_gettime).with(Process::CLOCK_MONOTONIC).and_return(20.0)
    end

    result = runner.run("Help", limits: { max_duration: 5 })

    expect(result.error).to be_a(Agents::ExecutionBudget::Exceeded)
    expect(result.usage.total_tokens).to eq(18)
    expect(request).to have_been_requested.once
  end

  it "does not depend on successfully installing observability callbacks" do
    request = stub_chat_sequence(tool_response, "Done")
    runner = described_class.with_agents(agent)
    runner.on_chat_created { raise "observability unavailable" }
    runner.on_tool_start { raise "logging unavailable" }

    result = runner.run("Help", max_turns: 1)

    expect(result.error).to be_a(Agents::Runner::MaxTurnsExceeded)
    expect(request).to have_been_requested.once
    expect(effects).to eq(["completed"])
  end

  it "does not reset the caller budget inside an agent tool" do
    child = Agents::Agent.new(name: "Child", model: "gpt-4o", tools: [tool])
    parent = Agents::Agent.new(name: "Parent", model: "gpt-4o", tools: [child.as_tool])
    request = stub_chat_sequence(
      { tool_calls: [{ name: "child", arguments: { input: "Help" } }] }, tool_response, "Done"
    )

    result = described_class.with_agents(parent).run("Help", max_turns: 2)

    expect(result.error).to be_a(Agents::Runner::MaxTurnsExceeded)
    expect(request).to have_been_requested.twice
    expect(effects).to eq(["completed"])
    expect(result.usage.total_tokens).to eq(50)
    expect(result.context[:execution_counts]).to include(model_calls: 2, tool_calls: 2)
  end

  it "keeps budgets isolated between calls on the same reusable runner" do
    request = stub_simple_chat("Done")
    runner = described_class.with_agents(agent)

    results = Array.new(2) { runner.run("Help", max_turns: 1) }

    expect(results).to all(be_success)
    expect(results.map { |result| result.context[:execution_counts][:model_calls] }).to eq([1, 1])
    expect(request).to have_been_requested.twice
  end

  it "disables hidden transport retries without changing global RubyLLM configuration" do
    retries = RubyLLM.config.max_retries
    request = stub_request(:post, "https://api.openai.com/v1/chat/completions")
              .to_return(status: 500, body: '{"error":{"message":"unavailable"}}',
                         headers: { "Content-Type" => "application/json" })

    result = described_class.with_agents(agent).run("Help")

    expect(result).to be_failed
    expect(request).to have_been_requested.once
    expect(RubyLLM.config.max_retries).to eq(retries)
    expect(result.context[:execution_counts][:model_calls]).to eq(1)
  end
end
