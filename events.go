package step

import "encoding/json"

// AssistantEventType represents normalized streaming event kinds.
type AssistantEventType string

const (
	EventStart         AssistantEventType = "start"
	EventTextStart     AssistantEventType = "text_start"
	EventTextDelta     AssistantEventType = "text_delta"
	EventTextEnd       AssistantEventType = "text_end"
	EventThinkingStart AssistantEventType = "thinking_start"
	EventThinkingDelta AssistantEventType = "thinking_delta"
	EventThinkingEnd   AssistantEventType = "thinking_end"
	EventToolCallStart AssistantEventType = "toolcall_start"
	EventToolCallDelta AssistantEventType = "toolcall_delta"
	EventToolCallEnd   AssistantEventType = "toolcall_end"
	EventDone          AssistantEventType = "done"
	EventError         AssistantEventType = "error"
)

// AssistantEvent is the streaming update emitted by providers.
type AssistantEvent struct {
	Type      AssistantEventType
	PartIndex int
	Delta     string

	ToolCall *ToolCallPart

	Partial *Message

	Reason StopReason
	Err    string
}

// StepEventType represents step-level lifecycle updates.
type StepEventType string

const (
	StepEventStart          StepEventType = "step_start"
	StepEventAssistant      StepEventType = "assistant_event"
	StepEventToolExecStart  StepEventType = "tool_exec_start"
	StepEventToolExecUpdate StepEventType = "tool_exec_update"
	StepEventToolExecEnd    StepEventType = "tool_exec_end"
	StepEventEnd            StepEventType = "step_end"
)

// StepEvent wraps assistant events and tool execution progress.
type StepEvent struct {
	Type StepEventType

	Assistant *AssistantEvent

	ToolCallID  string
	ToolName    string
	ToolArgs    json.RawMessage
	ToolPartial any
	ToolResult  *ToolResult

	Final *StepResult
}
