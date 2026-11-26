# Execution limits

`max_turns` defaults to 10 and counts admitted model calls, including calls made by agents used as tools and provider fallbacks. A nested agent retains its own three-call limit and also draws from its parent's remaining budget.

Callers can set optional limits for tools, accepted handoffs and elapsed seconds. These values are examples:

```ruby
result = runner.run(
  "Check this request",
  max_turns: 10,
  limits: { max_tool_calls: 12, max_handoffs: 3, max_duration: 60 }
)
```

Omitted tool, handoff and duration limits are unbounded. Counts must be non-negative integers and duration must be a finite, non-negative number. Each run exposes its counters in `result.context[:execution_counts]`.

Model-call exhaustion returns `Runner::MaxTurnsExceeded` in `result.error`. Tool, handoff and duration limits return `ExecutionBudget::Exceeded`. Completed tool results and token usage stay available if a later attempt is blocked. The denied tool receives an error result so its request remains linked to a result in the conversation history. Admission counts include calls reserved before a synchronous callback even if a deadline expires in that callback.

The SDK disables transport retries in a per-run RubyLLM context and counts retries through RubyLLM's configured model fallbacks. Custom provider clients should disable their own retries. The deadline stops new work and rejects a late final response. An in-flight network request or tool can still complete, so configure provider and tool timeouts separately.

Nested runs keep separate history and share only the caller's state and execution budget. Child token usage is added to the parent total once. The parent receives child model and chat observations for tracing. Child tool and lifecycle callbacks stay in the child run.
