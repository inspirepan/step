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

// ToolExecStartDelta signals tool execution start with the full call info.
type ToolExecStartDelta struct {
	Call ToolCallPart
}

func (ToolExecStartDelta) deltaKind() DeltaKind { return DeltaToolExec }

// StepStatusDelta reports step-level status updates.
type StepStatusDelta struct {
	Cancelled bool
}

func (StepStatusDelta) deltaKind() DeltaKind { return DeltaStep }
