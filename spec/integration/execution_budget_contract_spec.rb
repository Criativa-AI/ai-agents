# frozen_string_literal: true

require "webmock/rspec"

RSpec.describe Agents::Runner, "#run" do
  include OpenAITestHelper

  let(:endpoint) { "https://api.openai.com/v1/chat/completions" }
  let(:tool) do
    Class.new(Agents::Tool) do
      description "Record a step"
      parameter :label, type: :string

      def name
        "record_step"
      end

      def perform(context, label:)
        (context.state[:steps] ||= []) << label
        label
      end
    end.new
  end
  let(:agent) { Agents::Agent.new(name: "Owner", model: "gpt-4o", protocol: :chat_completions, tools: [tool]) }

  before do
    setup_openai_test_config
    WebMock.disable_net_connect!
  end

  def stub_turns(*bodies)
    stub_request(:post, endpoint).to_return(*bodies.map do |body|
      { status: 200, headers: { "Content-Type" => "application/json" }, body: body.to_json }
    end)
  end

  it "preserves the first result and records an error for a tool denied by the shared limit" do
    calls = [
      { id: "one", name: tool.name, arguments: { label: "first" } },
      { id: "two", name: tool.name, arguments: { label: "second" } }
    ]
    request = stub_turns(build_tool_call_response(calls))

    result = described_class.with_agents(agent).run("Record both steps", limits: { max_tool_calls: 1 })

    expect(result.error).to be_a(Agents::ExecutionBudget::Exceeded)
    expect(result.context[:state][:steps]).to eq(["first"])
    expect(result.context[:execution_counts]).to include(model_calls: 1, tool_calls: 1)
    expect(result.messages.select { |message| message[:role] == :tool }.map { |message| message[:tool_call_id] })
      .to eq(%w[one two])
    expect(request).to have_been_requested.once
  end

  it "shares the parent's model-call limit and counts child usage exactly once" do
    child = Agents::Agent.new(name: "Child", model: "gpt-4o", protocol: :chat_completions)
    wrapped = child.as_tool(name: "ask_child")
    owner = Agents::Agent.new(name: "Owner", model: "gpt-4o", protocol: :chat_completions, tools: [wrapped])
    calls = [{ id: "child", name: wrapped.name, arguments: { input: "Return a fixed result" } }]
    request = stub_turns(build_tool_call_response(calls), build_simple_response("child result"))
    observed = []
    runner = described_class.with_agents(owner)
    runner.on_llm_call_complete { |name, _model, _response, context| observed << [name, context.object_id] }

    result = runner.run("Call the child", max_turns: 2)

    expect(result.error).to be_a(described_class::MaxTurnsExceeded)
    expect(result.context[:execution_counts]).to include(model_calls: 2, tool_calls: 1)
    expect(result.usage.total_tokens).to eq(43)
    expect(observed.map(&:first)).to eq(%w[Owner Child])
    expect(observed.map(&:last).uniq.size).to eq(1)
    expect(request).to have_been_requested.twice
  end

  it "rejects a handoff before the acceptance hook when no handoffs remain" do
    target = Agents::Agent.new(name: "Specialist", model: "gpt-4o", protocol: :chat_completions)
    accepted = []
    agent.register_handoff(target, on_handoff: ->(*args) { accepted << args })
    handoff = { id: "handoff", name: "handoff_to_specialist", arguments: {} }
    request = stub_turns(build_tool_call_response([handoff]))

    result = described_class.with_agents(agent, target).run("Transfer", limits: { max_handoffs: 0 })

    expect(result.error).to be_a(Agents::ExecutionBudget::Exceeded)
    expect(result.context[:execution_counts]).to include(model_calls: 1, tool_calls: 1, handoffs: 0)
    expect(accepted).to be_empty
    expect(request).to have_been_requested.once
  end

  it "does not start a provider request after the deadline" do
    request = stub_turns(build_simple_response("unexpected"))

    result = described_class.with_agents(agent).run("Check limit", limits: { max_duration: 0 })

    expect(result.error).to be_a(Agents::ExecutionBudget::Exceeded)
    expect(result.context[:execution_counts]).to include(model_calls: 0)
    expect(request).not_to have_been_requested
  end

  it "does not retry a failed provider request outside the model-call count" do
    request = stub_request(:post, endpoint).to_return(
      status: 503, headers: { "Content-Type" => "application/json" }, body: '{"error":{"message":"unavailable"}}'
    )

    result = described_class.with_agents(agent).run("Check retries", max_turns: 1)

    expect(result.error).to be_a(RubyLLM::Error)
    expect(result.context[:execution_counts]).to include(model_calls: 1)
    expect(request).to have_been_requested.once
  end

  it "counts a configured fallback as another model call" do
    request = stub_request(:post, endpoint).to_return(
      status: 503, headers: { "Content-Type" => "application/json" }, body: '{"error":{"message":"unavailable"}}'
    )
    runner = described_class.with_agents(agent)
    runner.on_chat_created { |chat| chat.with_fallbacks("gpt-4o-mini") }

    result = runner.run("Check fallback budget", max_turns: 1)

    expect(result.error).to be_a(described_class::MaxTurnsExceeded)
    expect(result.context[:execution_counts]).to include(model_calls: 1)
    expect(request).to have_been_requested.once
  end

  it "rechecks the deadline after a synchronous observer before the next provider request" do
    now = 100.0
    allow(Process).to receive(:clock_gettime).and_call_original
    allow(Process).to receive(:clock_gettime).with(Process::CLOCK_MONOTONIC) { now }
    request = stub_turns(build_simple_response("unexpected"))
    runner = described_class.with_agents(agent)
    runner.on_agent_thinking { now += 2 }

    result = runner.run("Check observer deadline", limits: { max_duration: 1 })

    expect(result.error).to be_a(Agents::ExecutionBudget::Exceeded)
    expect(result.context[:execution_counts]).to include(model_calls: 1)
    expect(request).not_to have_been_requested
  end

  it "rechecks the deadline after a synchronous tool observer before running the tool" do
    now = 100.0
    allow(Process).to receive(:clock_gettime).and_call_original
    allow(Process).to receive(:clock_gettime).with(Process::CLOCK_MONOTONIC) { now }
    calls = [{ id: "late", name: tool.name, arguments: { label: "late effect" } }]
    request = stub_turns(build_tool_call_response(calls))
    runner = described_class.with_agents(agent)
    runner.on_tool_start { now += 2 }
    state = {}

    result = runner.run("Check tool deadline", context: { state: state }, limits: { max_duration: 1 })

    expect(result.error).to be_a(Agents::ExecutionBudget::Exceeded)
    expect(state).to be_empty
    expect(result.messages.any? { |message| message[:role] == :tool && message[:tool_call_id] == "late" }).to be(true)
    expect(request).to have_been_requested.once
  end
end
