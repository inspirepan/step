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

// PartType describes the kind of content in a part.
type PartType string

const (
	PartText     PartType = "text"
	PartThinking PartType = "thinking"
	PartImage    PartType = "image"
	PartToolCall PartType = "tool_call"
)

// Part is a structured message fragment.
type Part interface {
	partType() PartType
}

// TextPart represents text content.
type TextPart struct {
	Text string `json:"text"`
}

func (TextPart) partType() PartType { return PartText }

func (p TextPart) MarshalJSON() ([]byte, error) {
	type alias TextPart
	return json.Marshal(struct {
		Type PartType `json:"type"`
		alias
	}{PartText, alias(p)})
}

// ThinkingPart represents model reasoning content.
type ThinkingPart struct {
	ID string `json:"id,omitempty"`
	// Thinking is the text content or summary
	Thinking string `json:"thinking,omitempty"`
	// Signature for Claude and Gemini or reasoning.encrypted_content for OpenAI
	Signature string `json:"signature,omitempty"`
	// Format for OpenRouter's reasoning_detail.format
	Format string `json:"format,omitempty"`
	// ModelName identifies the source model for cross-model degradation
	ModelName string `json:"model_name,omitempty"`
}

func (ThinkingPart) partType() PartType { return PartThinking }

func (p ThinkingPart) MarshalJSON() ([]byte, error) {
	type alias ThinkingPart
	return json.Marshal(struct {
		Type PartType `json:"type"`
		alias
	}{PartThinking, alias(p)})
}

// ImagePart represents image content.
type ImagePart struct {
	MimeType string `json:"mime_type"`
	DataB64  string `json:"data_b64"`
}

func (ImagePart) partType() PartType { return PartImage }

func (p ImagePart) MarshalJSON() ([]byte, error) {
	type alias ImagePart
	return json.Marshal(struct {
		Type PartType `json:"type"`
		alias
	}{PartImage, alias(p)})
}

// ToolCallPart represents a tool call request.
type ToolCallPart struct {
	CallID   string          `json:"call_id"`
	Name     string          `json:"name"`
	ArgsJSON json.RawMessage `json:"args_json,omitempty"`
}

func (ToolCallPart) partType() PartType { return PartToolCall }

func (p ToolCallPart) MarshalJSON() ([]byte, error) {
	type alias ToolCallPart
	return json.Marshal(struct {
		Type PartType `json:"type"`
		alias
	}{PartToolCall, alias(p)})
}

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
	Parts     []Part `json:"parts,omitempty"`
	Timestamp int64  `json:"timestamp"`
}

func (AssistantMessage) role() Role { return RoleAssistant }

func (m AssistantMessage) MarshalJSON() ([]byte, error) {
	type alias AssistantMessage
	return json.Marshal(struct {
		Role Role `json:"role"`
		alias
	}{RoleAssistant, alias(m)})
}

// ToolMessage represents a tool execution result message.
type ToolMessage struct {
	CallID    string         `json:"call_id"`
	Name      string         `json:"name"`
	IsError   bool           `json:"is_error,omitempty"`
	Parts     []Part         `json:"parts,omitempty"`
	Timestamp int64          `json:"timestamp"`
	Details   map[string]any `json:"details,omitempty"`
}

func (ToolMessage) role() Role { return RoleTool }

func (m ToolMessage) MarshalJSON() ([]byte, error) {
	type alias ToolMessage
	return json.Marshal(struct {
		Role Role `json:"role"`
		alias
	}{RoleTool, alias(m)})
}

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

// UnmarshalPart decodes a JSON object into a concrete Part type.
func UnmarshalPart(data []byte) (Part, error) {
	var raw struct {
		Type PartType `json:"type"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}

	switch raw.Type {
	case PartText:
		var p TextPart
		if err := json.Unmarshal(data, &p); err != nil {
			return nil, err
		}
		return p, nil
	case PartThinking:
		var p ThinkingPart
		if err := json.Unmarshal(data, &p); err != nil {
			return nil, err
		}
		return p, nil
	case PartImage:
		var p ImagePart
		if err := json.Unmarshal(data, &p); err != nil {
			return nil, err
		}
		return p, nil
	case PartToolCall:
		var p ToolCallPart
		if err := json.Unmarshal(data, &p); err != nil {
			return nil, err
		}
		return p, nil
	default:
		return nil, fmt.Errorf("unknown part type: %s", raw.Type)
	}
}

func unmarshalParts(rawParts []json.RawMessage) ([]Part, error) {
	parts := make([]Part, 0, len(rawParts))
	for _, raw := range rawParts {
		p, err := UnmarshalPart(raw)
		if err != nil {
			return nil, err
		}
		parts = append(parts, p)
	}
	return parts, nil
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

func (m *ToolMessage) UnmarshalJSON(data []byte) error {
	type alias ToolMessage
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
		var m ToolMessage
		if err := json.Unmarshal(data, &m); err != nil {
			return nil, err
		}
		return m, nil
	default:
		return nil, fmt.Errorf("unknown role: %s", raw.Role)
	}
}
