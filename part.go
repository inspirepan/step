package step

import (
	"encoding/json"
	"fmt"
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
