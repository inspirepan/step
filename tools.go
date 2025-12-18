package step

import (
	"context"
	"encoding/json"
)

// ToolSpec is the declarative tool schema exposed to LLM.
type ToolSpec struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
	Parallel    bool           `json:"-"` // if true, tool can be executed in parallel, e.g. sub-agent, web_search, web_fetch and other read-only tools
}

// ToolCall is the normalized tool call.
type ToolCall struct {
	CallID   string
	Name     string
	ArgsJSON json.RawMessage
}

// ToolResult is the normalized tool execution result.
type ToolResult struct {
	CallID  string
	Name    string
	Parts   []Part
	IsError bool
	Details map[string]any // extra data, e.g. diff text for edit tool UI rendering
}

// Tool is an executable tool.
type Tool interface {
	Spec() ToolSpec
	Execute(ctx context.Context, call ToolCallPart) (ToolResult, error)
}
