package openrouter

import (
	"github.com/inspirepan/step"
	cc "github.com/inspirepan/step/providers/chatcompletion"
)

// ReasoningHandler handles OpenRouter's reasoning_details format for Claude and
// other reasoning models. OpenRouter uses a different format than standard
// Chat Completion API.
//
// See: https://openrouter.ai/docs/use-cases/reasoning-tokens
type ReasoningHandler struct {
	modelName   string
	parts       []step.ThinkingPart
	currentPart *step.ThinkingPart
}

// Ensure ReasoningHandler implements the interface
var _ cc.ReasoningHandler = (*ReasoningHandler)(nil)

// NewReasoningHandler creates a new OpenRouter reasoning handler.
func NewReasoningHandler(modelName string) *ReasoningHandler {
	return &ReasoningHandler{
		modelName: modelName,
		parts:     make([]step.ThinkingPart, 0, 1),
	}
}

// ConvertThinkingToExtra converts ThinkingParts to reasoning_details format.
// Each ThinkingPart is converted to reasoning_detail objects based on format:
// - anthropic-claude-v1: single reasoning.text with embedded signature
// - openai-responses-v1: reasoning.summary + separate reasoning.encrypted
// - other: reasoning.text + separate reasoning.encrypted
func (h *ReasoningHandler) ConvertThinkingToExtra(parts []step.ThinkingPart, targetModel string) (string, any, string) {
	var details []map[string]any
	var degradedText string

	for i, part := range parts {
		// Cross-model: degrade to text if models don't match
		if part.ModelName != "" && part.ModelName != targetModel {
			if part.Thinking != "" {
				degradedText += "<thinking>\n" + part.Thinking + "\n</thinking>\n"
			}
			continue
		}

		// Same model: convert to reasoning_details format based on format type
		switch part.Format {
		case "anthropic-claude-v1":
			// Claude format: single reasoning.text with embedded signature
			if part.Thinking != "" || part.Signature != "" {
				detail := map[string]any{
					"type":   "reasoning.text",
					"index":  i,
					"format": part.Format,
				}
				if part.Thinking != "" {
					detail["text"] = part.Thinking
				}
				if part.Signature != "" {
					detail["signature"] = part.Signature
				}
				if part.ID != "" {
					detail["id"] = part.ID
				}
				details = append(details, detail)
			}

		case "openai-responses-v1":
			// OpenAI format: reasoning.summary + separate reasoning.encrypted
			if part.Thinking != "" {
				detail := map[string]any{
					"type":    "reasoning.summary",
					"summary": part.Thinking,
					"format":  part.Format,
					"index":   i,
				}
				if part.ID != "" {
					detail["id"] = part.ID
				}
				details = append(details, detail)
			}
			if part.Signature != "" {
				details = append(details, map[string]any{
					"type":   "reasoning.encrypted",
					"data":   part.Signature,
					"format": part.Format,
					"index":  i,
				})
			}

		default:
			// Default: reasoning.text + separate reasoning.encrypted for Gemini and Grok
			if part.Thinking != "" {
				detail := map[string]any{
					"type":  "reasoning.text",
					"text":  part.Thinking,
					"index": i,
				}
				if part.Format != "" {
					detail["format"] = part.Format
				}
				if part.ID != "" {
					detail["id"] = part.ID
				}
				details = append(details, detail)
			}
			if part.Signature != "" {
				encryptedDetail := map[string]any{
					"type":  "reasoning.encrypted",
					"data":  part.Signature,
					"index": i,
				}
				if part.Format != "" {
					encryptedDetail["format"] = part.Format
				}
				if part.ID != "" {
					encryptedDetail["id"] = part.ID
				}
				details = append(details, encryptedDetail)
			}
		}
	}

	if len(details) == 0 {
		return "", nil, degradedText
	}

	return "reasoning_details", details, degradedText
}

// ExtractThinking extracts thinking from OpenRouter's reasoning_details delta.
// Accumulates text from reasoning.text/reasoning.summary, completes a ThinkingPart
// when reasoning.encrypted is received.
func (h *ReasoningHandler) ExtractThinking(delta map[string]any) (string, bool) {
	// OpenRouter uses reasoning_details array in delta
	reasoningDetails, ok := delta["reasoning_details"].([]any)
	if !ok || len(reasoningDetails) == 0 {
		return "", false
	}

	var allText string
	var isThinking bool
	for _, item := range reasoningDetails {
		detail, ok := item.(map[string]any)
		if !ok {
			continue
		}

		detailType, _ := detail["type"].(string)

		isThinking = true

		switch detailType {
		case "reasoning.text":
			// Initialize current part if needed
			if h.currentPart == nil {
				h.currentPart = &step.ThinkingPart{
					ModelName: h.modelName,
				}
			}

			// Extract fields
			if id, ok := detail["id"].(string); ok && id != "" {
				h.currentPart.ID = id
			}
			if format, ok := detail["format"].(string); ok && format != "" {
				h.currentPart.Format = format
			}
			if text, ok := detail["text"].(string); ok && text != "" {
				h.currentPart.Thinking += text
				allText += text
			}

			// Check for embedded signature (anthropic-claude-v1 format)
			if sig, ok := detail["signature"].(string); ok && sig != "" {
				h.currentPart.Signature = sig
				// Finalize this ThinkingPart and start a new one
				h.parts = append(h.parts, *h.currentPart)
				h.currentPart = nil
			}

		case "reasoning.summary":
			// Initialize current part if needed
			if h.currentPart == nil {
				h.currentPart = &step.ThinkingPart{
					ModelName: h.modelName,
				}
			}

			// Extract fields
			if id, ok := detail["id"].(string); ok && id != "" {
				h.currentPart.ID = id
			}
			if format, ok := detail["format"].(string); ok && format != "" {
				h.currentPart.Format = format
			}
			if summary, ok := detail["summary"].(string); ok && summary != "" {
				h.currentPart.Thinking += summary
				allText += summary
			}

		case "reasoning.encrypted":
			// Complete the current ThinkingPart with signature
			if h.currentPart == nil {
				h.currentPart = &step.ThinkingPart{
					ModelName: h.modelName,
				}
			}

			// Extract fields
			if id, ok := detail["id"].(string); ok && id != "" {
				h.currentPart.ID = id
			}

			if data, ok := detail["data"].(string); ok && data != "" {
				h.currentPart.Signature = data
			}
			if format, ok := detail["format"].(string); ok && format != "" {
				h.currentPart.Format = format
			}

			// Finalize this ThinkingPart and start a new one
			h.parts = append(h.parts, *h.currentPart)
			h.currentPart = nil
		}
	}

	if allText != "" {
		return allText, true
	}
	return "", isThinking
}

// FlushThinking returns accumulated thinking as ThinkingParts.
// For models that don't return encrypted signatures, the current part
// (without signature) will be included.
func (h *ReasoningHandler) FlushThinking() []step.ThinkingPart {
	// Include current part if it has content (for models without encrypted signature)
	if h.currentPart != nil && h.currentPart.Thinking != "" {
		h.parts = append(h.parts, *h.currentPart)
		h.currentPart = nil
	}

	if len(h.parts) == 0 {
		return nil
	}

	result := h.parts
	h.parts = make([]step.ThinkingPart, 0, 1)
	return result
}
