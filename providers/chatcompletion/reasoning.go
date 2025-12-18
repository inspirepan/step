package chatcompletion

import "github.com/inspirepan/step"

// ReasoningField is the key for reasoning content in provider-specific extra fields.
const ReasoningField = "reasoning"

// ReasoningHandler abstracts provider-specific reasoning/thinking handling.
// Standard Chat Completion API has no reasoning support
type ReasoningHandler interface {
	// --- Input conversion (for building requests) ---

	// ConvertThinkingToExtra converts ThinkingParts to provider-specific extra field format.
	// Takes all ThinkingParts from a message and converts them together.
	// Returns:
	// - key: the extra field key (e.g. "reasoning_details" for OpenRouter), empty if not supported
	// - value: the extra field value (e.g. array of reasoning_detail objects)
	// - degradedText: text to prepend to content if cross-model or unsupported
	ConvertThinkingToExtra(parts []step.ThinkingPart, targetModel string) (key string, value any, degradedText string)

	// --- Output extraction (for parsing responses) ---

	// ExtractThinking extracts thinking content from a streaming delta.
	// Returns the thinking text if present, empty string otherwise.
	ExtractThinking(delta map[string]any) (text string, isThinking bool)

	// FlushThinking returns accumulated thinking as ThinkingParts.
	FlushThinking() []step.ThinkingPart
}

// NoOpReasoningHandler is the default handler that does nothing with reasoning.
// Used for standard OpenAI Chat Completion API which doesn't support reasoning.
type NoOpReasoningHandler struct{}

func (h *NoOpReasoningHandler) ConvertThinkingToExtra(parts []step.ThinkingPart, _ string) (string, any, string) {
	return "", nil, ""
}

func (h *NoOpReasoningHandler) ExtractThinking(_ map[string]any) (string, bool) {
	return "", false
}

func (h *NoOpReasoningHandler) FlushThinking() []step.ThinkingPart {
	return nil
}

// DefaultReasoningHandler handles reasoning_content (used by some OpenAI-compatible APIs).
type DefaultReasoningHandler struct {
	modelName           string
	accumulatedThinking []string
}

func NewDefaultReasoningHandler(modelName string) *DefaultReasoningHandler {
	return &DefaultReasoningHandler{
		modelName:           modelName,
		accumulatedThinking: make([]string, 0),
	}
}

func (h *DefaultReasoningHandler) ConvertThinkingToExtra(parts []step.ThinkingPart, targetModel string) (string, any, string) {
	var reasoning string
	var degradedText string

	for _, p := range parts {
		if p.Thinking == "" {
			continue
		}
		// Cross-model: degrade to text if models don't match
		if p.ModelName != "" && p.ModelName != targetModel {
			degradedText += p.Thinking
			continue
		}
		// Same model: accumulate for reasoning field
		reasoning += p.Thinking
	}

	if reasoning == "" {
		return "", nil, degradedText
	}

	return ReasoningField, reasoning, degradedText
}

func (h *DefaultReasoningHandler) ExtractThinking(delta map[string]any) (string, bool) {
	if reasoning, ok := delta[ReasoningField].(string); ok && reasoning != "" {
		h.accumulatedThinking = append(h.accumulatedThinking, reasoning)
		return reasoning, true
	}
	return "", false
}

func (h *DefaultReasoningHandler) FlushThinking() []step.ThinkingPart {
	if len(h.accumulatedThinking) == 0 {
		return nil
	}
	content := ""
	for _, t := range h.accumulatedThinking {
		content += t
	}
	h.accumulatedThinking = h.accumulatedThinking[:0]
	return []step.ThinkingPart{
		{
			Thinking:  content,
			ModelName: h.modelName,
		},
	}
}
