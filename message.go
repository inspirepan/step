package step

import (
	"encoding/json"
	"fmt"
)

// Role is the speaker role.
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// Message is the canonical conversation unit.
type Message interface {
	role() Role
}

// UserMessage represents a user input message.
type UserMessage struct {
	Parts     []Part `json:"parts,omitempty"`
	Timestamp int64  `json:"timestamp"`
}

func (UserMessage) role() Role { return RoleUser }

func (m UserMessage) MarshalJSON() ([]byte, error) {
	type alias UserMessage
	return json.Marshal(struct {
		Role Role `json:"role"`
		alias
	}{RoleUser, alias(m)})
}

// AssistantMessage represents an assistant response message.
type AssistantMessage struct {
	Parts      []Part     `json:"parts,omitempty"`
	Timestamp  int64      `json:"timestamp"`
	Usage      *Usage     `json:"usage,omitempty"`
	StopReason StopReason `json:"stop_reason,omitempty"`
}

func (AssistantMessage) role() Role { return RoleAssistant }

func (m AssistantMessage) MarshalJSON() ([]byte, error) {
	type alias AssistantMessage
	return json.Marshal(struct {
		Role Role `json:"role"`
		alias
	}{RoleAssistant, alias(m)})
}

// ToolResultMessage represents a tool execution result message.
type ToolResultMessage struct {
	CallID    string         `json:"call_id"`
	Name      string         `json:"name"`
	IsError   bool           `json:"is_error,omitempty"`
	Parts     []Part         `json:"parts,omitempty"`
	Timestamp int64          `json:"timestamp"`
	Details   map[string]any `json:"details,omitempty"`
}

func (ToolResultMessage) role() Role { return RoleTool }

func (m ToolResultMessage) MarshalJSON() ([]byte, error) {
	type alias ToolResultMessage
	return json.Marshal(struct {
		Role Role `json:"role"`
		alias
	}{RoleTool, alias(m)})
}

// ToolMessage is kept for backward compatibility.
type ToolMessage = ToolResultMessage

// StopReason explains why generation stopped.
type StopReason string

const (
	StopStop    StopReason = "stop"
	StopLength  StopReason = "length"
	StopToolUse StopReason = "tool_use"
	StopError   StopReason = "error"
	StopAborted StopReason = "aborted"
)

// Usage reports token accounting.
type Usage struct {
	InputTokens      int `json:"input_tokens"`
	OutputTokens     int `json:"output_tokens"`
	CachedReadTokens int `json:"cached_read_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

func (m *UserMessage) UnmarshalJSON(data []byte) error {
	type alias UserMessage
	aux := &struct {
		Parts []json.RawMessage `json:"parts,omitempty"`
		*alias
	}{alias: (*alias)(m)}
	if err := json.Unmarshal(data, aux); err != nil {
		return err
	}
	parts, err := unmarshalParts(aux.Parts)
	if err != nil {
		return err
	}
	m.Parts = parts
	return nil
}

func (m *AssistantMessage) UnmarshalJSON(data []byte) error {
	type alias AssistantMessage
	aux := &struct {
		Parts []json.RawMessage `json:"parts,omitempty"`
		*alias
	}{alias: (*alias)(m)}
	if err := json.Unmarshal(data, aux); err != nil {
		return err
	}
	parts, err := unmarshalParts(aux.Parts)
	if err != nil {
		return err
	}
	m.Parts = parts
	return nil
}

func (m *ToolResultMessage) UnmarshalJSON(data []byte) error {
	type alias ToolResultMessage
	aux := &struct {
		Parts []json.RawMessage `json:"parts,omitempty"`
		*alias
	}{alias: (*alias)(m)}
	if err := json.Unmarshal(data, aux); err != nil {
		return err
	}
	parts, err := unmarshalParts(aux.Parts)
	if err != nil {
		return err
	}
	m.Parts = parts
	return nil
}

// UnmarshalMessage decodes a JSON object into a concrete Message type.
func UnmarshalMessage(data []byte) (Message, error) {
	var raw struct {
		Role Role `json:"role"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}

	switch raw.Role {
	case RoleUser:
		var m UserMessage
		if err := json.Unmarshal(data, &m); err != nil {
			return nil, err
		}
		return m, nil
	case RoleAssistant:
		var m AssistantMessage
		if err := json.Unmarshal(data, &m); err != nil {
			return nil, err
		}
		return m, nil
	case RoleTool:
		var m ToolResultMessage
		if err := json.Unmarshal(data, &m); err != nil {
			return nil, err
		}
		return m, nil
	default:
		return nil, fmt.Errorf("unknown role: %s", raw.Role)
	}
}
