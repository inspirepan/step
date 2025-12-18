package step

import (
	"context"
	"sync"
	"time"
)

type indexedCall struct {
	idx  int
	call ToolCall
}

func (s *stepStream) executeTools(calls []ToolCall, tools []Tool) ([]ToolResult, bool) {
	if len(calls) == 0 {
		return nil, false
	}

	toolMap := map[string]Tool{}
	for _, t := range tools {
		spec := t.Spec()
		toolMap[spec.Name] = t
	}

	// separate serial and parallel calls, preserving original indices
	var serialCalls, parallelCalls []indexedCall
	for i, call := range calls {
		tool, ok := toolMap[call.Name]
		if !ok || !tool.Spec().Parallel {
			serialCalls = append(serialCalls, indexedCall{idx: i, call: call})
		} else {
			parallelCalls = append(parallelCalls, indexedCall{idx: i, call: call})
		}
	}

	results := make([]ToolResult, len(calls))
	cancelled := false

	// phase 1: execute serial tools sequentially
	for _, ic := range serialCalls {
		if s.ctx.Err() != nil {
			cancelled = true
			results[ic.idx] = interruptedToolResult(ic.call)
			continue
		}
		results[ic.idx] = s.executeSingleTool(ic.call, toolMap)
	}

	// phase 2: execute parallel tools concurrently
	if cancelled || s.ctx.Err() != nil {
		// fill all parallel calls with interrupted results
		for _, ic := range parallelCalls {
			results[ic.idx] = interruptedToolResult(ic.call)
		}
		return results, true
	}

	if len(parallelCalls) > 0 {
		s.executeParallelTools(parallelCalls, toolMap, results)
		if s.ctx.Err() != nil {
			cancelled = true
		}
	}

	return results, cancelled
}

func (s *stepStream) executeSingleTool(call ToolCall, toolMap map[string]Tool) ToolResult {
	tool, ok := toolMap[call.Name]
	if !ok {
		s.addError(ErrToolNotFound)
		return toolNotFoundResult(call)
	}

	s.emit(StepEvent{Type: StepEventToolExecStart, ToolCallID: call.CallID, ToolName: call.Name, ToolArgs: call.ArgsJSON})

	toolCtx, cancel := context.WithCancel(s.ctx)
	res, err := tool.Execute(toolCtx, call)
	cancel()
	if err != nil {
		res = errorToolResult(call, err)
	}

	s.emit(StepEvent{Type: StepEventToolExecEnd, ToolCallID: call.CallID, ToolName: call.Name, ToolArgs: call.ArgsJSON, ToolResult: &res})
	return res
}

func (s *stepStream) executeParallelTools(calls []indexedCall, toolMap map[string]Tool, results []ToolResult) {
	type item struct {
		idx int
		res ToolResult
	}

	out := make(chan item, len(calls))
	var wg sync.WaitGroup

	for _, ic := range calls {
		wg.Go(func() {
			select {
			case <-s.ctx.Done():
				out <- item{idx: ic.idx, res: interruptedToolResult(ic.call)}
				return
			default:
			}

			res := s.executeSingleTool(ic.call, toolMap)
			out <- item{idx: ic.idx, res: res}
		})
	}

	wg.Wait()
	close(out)

	for it := range out {
		results[it.idx] = it.res
	}
}

func collectToolSpecs(tools []Tool) []ToolSpec {
	specs := make([]ToolSpec, 0, len(tools))
	for _, t := range tools {
		specs = append(specs, t.Spec())
	}
	return specs
}

func extractToolCalls(msg AssistantMessage) []ToolCall {
	var calls []ToolCall
	for _, part := range msg.Parts {
		tc, ok := part.(ToolCallPart)
		if !ok {
			continue
		}
		calls = append(calls, ToolCall(tc))
	}
	return calls
}

func interruptedToolResult(call ToolCall) ToolResult {
	return ToolResult{
		CallID:  call.CallID,
		Name:    call.Name,
		IsError: true,
		Parts:   []Part{TextPart{Text: "user interrupted the tool call"}},
	}
}

func toolNotFoundResult(call ToolCall) ToolResult {
	return ToolResult{
		CallID:  call.CallID,
		Name:    call.Name,
		IsError: true,
		Parts:   []Part{TextPart{Text: "tool not found"}},
	}
}

func errorToolResult(call ToolCall, err error) ToolResult {
	return ToolResult{
		CallID:  call.CallID,
		Name:    call.Name,
		IsError: true,
		Parts:   []Part{TextPart{Text: err.Error()}},
	}
}

func toolResultsToMessages(results []ToolResult) []Message {
	msgs := make([]Message, 0, len(results))
	now := time.Now().UnixMilli()
	for _, res := range results {
		msgs = append(msgs, ToolMessage{
			CallID:    res.CallID,
			Name:      res.Name,
			IsError:   res.IsError,
			Parts:     res.Parts,
			Timestamp: now,
			Details:   res.Details,
		})
	}
	return msgs
}
