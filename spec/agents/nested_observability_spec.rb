# frozen_string_literal: true

require "webmock/rspec"

RSpec.describe Agents::Runner do
  include OpenAITestHelper

  let(:child) { Agents::Agent.new(name: "Child", model: "gpt-4o") }
  let(:parent) { Agents::Agent.new(name: "Parent", model: "gpt-4o", tools: [child.as_tool]) }
  let(:runner) { described_class.with_agents(parent) }
  let(:completions) { [] }
  let(:message_hooks) { [] }
  let(:lifecycle) { [] }
  let(:contexts) { [] }
  let(:child_call) { { tool_calls: [{ name: "child", arguments: { input: "Help" } }] } }

  before do
    setup_openai_test_config
    disable_net_connect!
    runner.on_run_start do |name, _input, context|
      lifecycle << [:start, name]
      contexts << context
    end
    runner.on_run_complete { |name| lifecycle << [:complete, name] }
    runner.on_tool_start { |name| lifecycle << [:tool_start, name] }
    runner.on_tool_complete { |name| lifecycle << [:tool_complete, name] }
    runner.on_llm_call_complete do |name, _model, response, context|
      completions << [name, response.input_tokens + response.output_tokens]
      contexts << context
    end
    runner.on_chat_created do |chat, name, _model, context|
      contexts << context
      chat.on_end_message { |message| message_hooks << name if message.role == :assistant }
    end
  end

  after { allow_net_connect! }

  it "reports each child response once while retaining the owning run's context and lifecycle" do
    request = stub_chat_sequence(child_call, "Child answer", "Parent answer")

    result = runner.run("Help", context: { correlation_id: "run-1" })

    expect(result).to be_success
    expect(request).to have_been_requested.times(3)
    expect(completions.map(&:first)).to eq(%w[Parent Child Parent])
    expect(completions.sum(&:last)).to eq(result.usage.total_tokens)
    expect(message_hooks).to eq(%w[Parent Child Parent])
    expect(contexts.map(&:object_id).uniq.size).to eq(1)
    expect(contexts.first.context).to include(correlation_id: "run-1", current_agent: "Parent")
    expect(lifecycle).to eq(
      [[:start, "Parent"], [:tool_start, "child"], [:tool_complete, "child"], [:complete, "Parent"]]
    )
  end

  it "keeps already observed usage when a nested run exhausts the shared budget" do
    request = stub_chat_sequence(child_call, "Child answer", "Unreachable")

    result = runner.run("Help", max_turns: 2)

    expect(result.error).to be_a(Agents::Runner::MaxTurnsExceeded)
    expect(request).to have_been_requested.twice
    expect(completions.map(&:first)).to eq(%w[Parent Child])
    expect(completions.sum(&:last)).to eq(result.usage.total_tokens)
    expect(lifecycle.count { |event| event.first == :complete }).to eq(1)
  end

  it "preserves execution and other observers when one completion callback fails" do
    stub_chat_sequence(child_call, "Child answer", "Parent answer")
    runner.on_llm_call_complete { raise "observer unavailable" }

    result = runner.run("Help")

    expect(result).to be_success
    expect(completions.map(&:first)).to eq(%w[Parent Child Parent])
    expect(message_hooks).to eq(%w[Parent Child Parent])
  end

  context "with two nested agent tools" do
    let(:grandchild) { Agents::Agent.new(name: "Grandchild", model: "gpt-4o") }
    let(:child) { Agents::Agent.new(name: "Child", model: "gpt-4o", tools: [grandchild.as_tool]) }

    it "forwards observations through both levels without duplicate events or totals" do
      grandchild_call = { tool_calls: [{ name: "grandchild", arguments: { input: "Help" } }] }
      request = stub_chat_sequence(child_call, grandchild_call, "Grandchild answer", "Child answer", "Parent answer")

      result = runner.run("Help")

      expect(result).to be_success
      expect(request).to have_been_requested.times(5)
      expect(completions.map(&:first)).to eq(%w[Parent Child Grandchild Child Parent])
      expect(message_hooks).to eq(%w[Parent Child Grandchild Child Parent])
      expect(completions.sum(&:last)).to eq(result.usage.total_tokens)
      expect(contexts.map(&:object_id).uniq.size).to eq(1)
      expect(lifecycle.count { |event| event.first == :complete }).to eq(1)
    end
  end
end
