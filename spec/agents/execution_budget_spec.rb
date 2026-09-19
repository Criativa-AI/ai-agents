# frozen_string_literal: true

require "spec_helper"

RSpec.describe Agents::ExecutionBudget do
  it "does not allow concurrent callers to exceed a shared budget" do
    budget = described_class.new(max_tool_calls: 1)
    results = Array.new(8) do
      Thread.new do
        budget.consume!(:tool_calls)
        :accepted
      rescue described_class::Exceeded
        :blocked
      end
    end.map(&:value)

    expect(results.count(:accepted)).to eq(1)
    expect(results.count(:blocked)).to eq(7)
    expect(budget.snapshot[:tool_calls]).to eq(1)
  end

  it "does not consume parent capacity when the child has no remaining capacity" do
    parent = described_class.new(max_model_calls: 2)
    child = described_class.new(max_model_calls: 0, parent: parent)

    expect { child.consume!(:model_calls) }.to raise_error(Agents::Runner::MaxTurnsExceeded)
    expect(parent.snapshot[:model_calls]).to eq(0)
  end

  [-1, 1.5, "10"].each do |limit|
    it "rejects invalid tool limit #{limit.inspect}" do
      expect { described_class.new(max_tool_calls: limit) }.to raise_error(ArgumentError)
    end
  end

  [-1, Float::INFINITY, Float::NAN, "10"].each do |duration|
    it "rejects invalid duration #{duration.inspect}" do
      expect { described_class.new(max_duration: duration) }.to raise_error(ArgumentError)
    end
  end
end
