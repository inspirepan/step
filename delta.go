package step

// DeltaKind describes the kind of a MessageDelta.
type DeltaKind string

const (
	DeltaStep     DeltaKind = "step"
	DeltaThinking DeltaKind = "thinking"
	DeltaText     DeltaKind = "text"
	DeltaToolCall DeltaKind = "tool_call"
	DeltaToolExec DeltaKind = "tool_exec"
)

// MessageDelta is a streaming-only update.
// It must never be appended into the conversation history.
type MessageDelta interface {
	deltaKind() DeltaKind
}

// ThinkingDelta streams reasoning/thinking content.
type ThinkingDelta struct {
	ID        string
	Delta     string
	Signature string
}

func (ThinkingDelta) deltaKind() DeltaKind { return DeltaThinking }

// TextDelta streams user-visible assistant text.
type TextDelta struct {
	Delta string
}

func (TextDelta) deltaKind() DeltaKind { return DeltaText }

// ToolCallDelta streams tool call construction.
type ToolCallDelta struct {
	CallID    string
	Name      string
	ArgsDelta string
}

func (ToolCallDelta) deltaKind() DeltaKind { return DeltaToolCall }

type ToolExecStage string

const (
	ToolExecStart ToolExecStage = "start"
	ToolExecEnd   ToolExecStage = "end"
)

// ToolExecDelta reports tool execution status.
type ToolExecDelta struct {
	CallID string
	Name   string
	Stage  ToolExecStage
}

func (ToolExecDelta) deltaKind() DeltaKind { return DeltaToolExec }

// StepStatusDelta reports step-level status updates.
type StepStatusDelta struct {
	Cancelled bool
}

func (StepStatusDelta) deltaKind() DeltaKind { return DeltaStep }
