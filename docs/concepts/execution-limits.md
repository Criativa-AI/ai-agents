# Execution limits

`max_turns` bounds actual model invocations, including RubyLLM's recursive tool
loop. It is checked before the next provider call. The default remains 10.
Every returned model response contributes to `result.usage` and emits one
`on_llm_call_complete` event, including responses that request a handoff.

Applications can additionally bound tools, accepted handoffs and elapsed time:

```ruby
runner.run(
  "Help with this request",
  max_turns: 10,
  limits: { max_tool_calls: 12, max_handoffs: 3, max_duration: 60 }
)
```

These additional values are examples, not defaults. Applications should select
them from their supported workflows and measured latency. Omitted additional
limits are unbounded. Counters are per run and available in
`result.context[:execution_counts]`. Invalid limits raise `ArgumentError` before
execution. Limits use monotonic elapsed time, not wall-clock dates.

## Failure and side effects

- Exhausting model calls produces `Runner::MaxTurnsExceeded` in `result.error`.
- Exhausting tools, handoffs or time produces `ExecutionBudget::Exceeded`.
- A response can request several tools; each attempted call consumes capacity.
- Once a handoff is accepted, remaining source tools in that batch are not
  executed. They receive an explicit skipped result, and the selected handoff
  remains the first accepted one.
- Successful earlier tool results and token usage remain in the run when a later
  action is blocked. Blocking does not roll back external effects.
- An agent used as a tool has its own three-call limit and shares the caller's
  remaining budget. Nested budget failures propagate instead of becoming an
  ordinary tool answer. Nested token usage is included once in the caller total.
- Observability callback failures cannot disable the budget. Admission checks
  are installed by the runtime, separately from best-effort callbacks.

## Network and deadline semantics

The runtime uses an isolated RubyLLM context with transport retries disabled.
This prevents hidden retries from bypassing the model-call count; it does not
change the application's global RubyLLM configuration. Custom providers must
also avoid hidden retries. Network failures are returned to the application.

Duration is an admission deadline: no new model or tool action starts after
expiry, and late final responses produce an error. It does not interrupt a tool
or HTTP request already in flight. Configure provider and external-tool timeouts
as well. Never assume a timed-out mutating request did not reach its destination.

## Compatibility

Handoff relationship schemas, metadata, hooks and callback signatures remain
available. Callback frequency and usage totals now reflect all model responses,
rather than only the final response of a RubyLLM tool loop. Consumers that relied
on undercounted totals must update their expectations.

This change does not provide durable task state, external-operation deduplication,
business-result validation, or proof that a human has taken over a conversation.

## Compatibility and nested observations

This release requires RubyLLM 1.15.0, the version verified by the fork and Captain.
RubyLLM 1.14 lacks the public admission hooks used by RuntimeChat.

Agent tools bridge `llm_call_complete` and `chat_created` to the owning run.
Observers receive the child's agent/model/response (or chat), with the owning
RunContext for correlation. Chat observers can install message hooks on child
chats, including the SDK tracing adapter. Child conversation history remains
isolated. Child lifecycle and tool callbacks are not replayed into the parent's
lifecycle: doing so could close its root span or overwrite its active tool.
Nested usage is added once to the aggregate result; completion events report
each response once, including responses before budget exhaustion.
