package chatcompletion

import (
	"context"
	"encoding/json"
	"io"
	"sort"
	"sync"
	"time"

	"github.com/inspirepan/step"
	"github.com/inspirepan/step/providers/base"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/ssestream"
)

// Stream implements step.ProviderStream for OpenAI Chat Completions API.
type Stream struct {
	providerName string
	modelName    string
	stream       *ssestream.Stream[openai.ChatCompletionChunk]
	debug        *base.DebugLogger

	reasoningHandler ReasoningHandler

	mu sync.Mutex

	done  bool
	err   error

	pending []step.ProviderUpdate

	// Accumulators
	textContent []string
	toolCalls   map[int]*toolCallAccumulator

	stopReason step.StopReason
	usage      *step.Usage
	parts      []step.Part
}

type toolCallAccumulator struct {
	id      string
	name    string
	argsStr string
}

func NewStream(
	providerName string,
	modelName string,
	stream *ssestream.Stream[openai.ChatCompletionChunk],
	handler ReasoningHandler,
	debug *base.DebugLogger,
) *Stream {
	if handler == nil {
		handler = &NoOpReasoningHandler{}
	}
	return &Stream{
		providerName:     providerName,
		modelName:        modelName,
		stream:           stream,
		debug:            debug,
		reasoningHandler: handler,
		toolCalls:        make(map[int]*toolCallAccumulator),
	}
}

func (s *Stream) Next(ctx context.Context) (step.ProviderUpdate, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.pending) > 0 {
		return s.dequeue()
	}
	if s.done {
		return nil, io.EOF
	}
	if s.err != nil {
		return nil, s.err
	}

	for {
		select {
		case <-ctx.Done():
			// If the user cancels mid-stream, finalize a partial message so callers can
			// still consume a coherent AssistantMessage (then Step will return ctx.Err()).
			if !s.done {
				s.finalize()
				if len(s.pending) > 0 {
					return s.dequeue()
				}
			}
			return nil, io.EOF
		default:
		}

		if !s.stream.Next() {
			if err := s.stream.Err(); err != nil {
				s.err = err
				return nil, s.err
			}
			s.finalize()
			if len(s.pending) > 0 {
				return s.dequeue()
			}
			return nil, io.EOF
		}

		chunk := s.stream.Current()
		s.processChunk(chunk)
		if len(s.pending) > 0 {
			return s.dequeue()
		}
	}
}

func (s *Stream) Close() error {
	if s.debug != nil {
		_ = s.debug.Close()
	}
	return s.stream.Close()
}

func (s *Stream) enqueue(up step.ProviderUpdate) {
	s.pending = append(s.pending, up)
}

func (s *Stream) dequeue() (step.ProviderUpdate, error) {
	up := s.pending[0]
	s.pending = s.pending[1:]

	if s.debug != nil {
		rec := base.NewDebugRecord("update", up)
		rec.Provider = s.providerName
		rec.Model = s.modelName
		_ = s.debug.Log(rec)
	}

	return up, nil
}

func (s *Stream) processChunk(chunk openai.ChatCompletionChunk) {
	if s.debug != nil {
		rec := base.NewDebugRecord("chunk", chunk.RawJSON())
		rec.Provider = s.providerName
		rec.Model = s.modelName
		_ = s.debug.Log(rec)
	}

	// Usage
	if chunk.Usage.TotalTokens > 0 {
		s.usage = &step.Usage{
			InputTokens:  int(chunk.Usage.PromptTokens),
			OutputTokens: int(chunk.Usage.CompletionTokens),
			TotalTokens:  int(chunk.Usage.TotalTokens),
		}
		if chunk.Usage.PromptTokensDetails.CachedTokens > 0 {
			s.usage.CachedReadTokens = int(chunk.Usage.PromptTokensDetails.CachedTokens)
		}
	}

	if len(chunk.Choices) == 0 {
		return
	}

	choice := chunk.Choices[0]
	delta := choice.Delta

	if choice.FinishReason != "" {
		s.stopReason = mapFinishReason(string(choice.FinishReason))
	}

	// Thinking (may be interleaved with text/tool calls in the same chunk)
	if s.reasoningHandler != nil {
		deltaMap := deltaToMap(delta)
		if text, isThinking := s.reasoningHandler.ExtractThinking(deltaMap); isThinking {
			// Some providers (e.g. OpenRouter+Gemini) may emit reasoning.encrypted with no text.
			if text != "" {
				s.enqueue(step.ProviderDeltaUpdate{Delta: step.ThinkingDelta{Delta: text}})
			}
			// Do not return: the same chunk can also include content/tool_calls.
		}
	}

	// Text (may be interleaved with tool calls)
	if delta.Content != "" {
		s.textContent = append(s.textContent, delta.Content)
		s.enqueue(step.ProviderDeltaUpdate{Delta: step.TextDelta{Delta: delta.Content}})
		// Do not return: the same chunk can also include tool_calls.
	}

	// Tool calls
	for _, tc := range delta.ToolCalls {
		idx := int(tc.Index)
		if _, exists := s.toolCalls[idx]; !exists {
			s.toolCalls[idx] = &toolCallAccumulator{}
		}
		acc := s.toolCalls[idx]
		if tc.ID != "" {
			acc.id = tc.ID
		}
		if tc.Function.Name != "" {
			acc.name = tc.Function.Name
		}
		if tc.Function.Arguments != "" {
			acc.argsStr += tc.Function.Arguments
			s.enqueue(step.ProviderDeltaUpdate{Delta: step.ToolCallDelta{CallID: acc.id, Name: acc.name, ArgsDelta: tc.Function.Arguments}})
		}
	}
}

func (s *Stream) finalize() {
	s.done = true

	if s.stopReason == "" {
		s.stopReason = step.StopStop
	}

	// Fixed final assembly order:
	// 1) thinking parts (always included if present)
	// 2) user-visible content parts (text today; future: text+image order)
	// 3) tool calls
	if thinkingParts := s.reasoningHandler.FlushThinking(); len(thinkingParts) > 0 {
		for _, part := range thinkingParts {
			s.parts = append(s.parts, part)
		}
	}
	// Text
	s.flushText()
	// Tool calls (stable by tool index)
	if len(s.toolCalls) > 0 {
		idxs := make([]int, 0, len(s.toolCalls))
		for idx := range s.toolCalls {
			idxs = append(idxs, idx)
		}
		sort.Ints(idxs)
		for _, idx := range idxs {
			acc := s.toolCalls[idx]
			if acc == nil || acc.id == "" || acc.name == "" {
				continue
			}
			s.parts = append(s.parts, step.ToolCallPart{
				CallID:   acc.id,
				Name:     acc.name,
				ArgsJSON: json.RawMessage(acc.argsStr),
			})
		}
	}

	msg := step.AssistantMessage{
		Parts:      s.parts,
		Timestamp:  time.Now().UnixMilli(),
		Usage:      s.usage,
		StopReason: s.stopReason,
	}
	s.enqueue(step.ProviderMessageUpdate{Message: msg})
}

func (s *Stream) flushText() {
	if len(s.textContent) == 0 {
		return
	}
	text := ""
	for _, t := range s.textContent {
		text += t
	}
	s.parts = append(s.parts, step.TextPart{Text: text})
	s.textContent = nil
}

func mapFinishReason(reason string) step.StopReason {
	switch reason {
	case "stop":
		return step.StopStop
	case "length":
		return step.StopLength
	case "tool_calls":
		return step.StopToolUse
	default:
		return step.StopStop
	}
}

func deltaToMap(delta openai.ChatCompletionChunkChoiceDelta) map[string]any {
	var m map[string]any
	_ = json.Unmarshal([]byte(delta.RawJSON()), &m)
	return m
}

var _ step.ProviderStream = (*Stream)(nil)
