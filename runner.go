package step

import (
	"context"
	"errors"
	"io"
	"time"
)

func runStep(ctx context.Context, req StepRequest, cfg stepConfig) (StepResult, error) {
	if req.Provider == nil {
		return nil, ErrNoProvider
	}

	emitter := cfg.stepEmitter

	providerReq := ProviderRequest{
		SystemPrompt: req.SystemPrompt,
		History:      req.History,
		Tools:        collectToolSpecs(req.Tools),
	}

	stream, err := req.Provider.Stream(ctx, providerReq)
	if err != nil {
		return nil, err
	}
	defer stream.Close()

	var assistantMsg AssistantMessage
	hasAssistantMsg := false

	for {
		up, nextErr := stream.Next(ctx)
		if nextErr != nil {
			if errors.Is(nextErr, io.EOF) {
				// Some providers may return a final update along with io.EOF.
				if up != nil {
					msg, ok, err := handleProviderUpdate(up, emitter)
					if err != nil {
						return nil, err
					}
					if ok {
						assistantMsg = msg
						hasAssistantMsg = true
					}
				}
				break
			}
			return nil, nextErr
		}
		msg, ok, err := handleProviderUpdate(up, emitter)
		if err != nil {
			return nil, err
		}
		if ok {
			assistantMsg = msg
			hasAssistantMsg = true
		}
	}

	if !hasAssistantMsg {
		return nil, errors.New("step: provider stream finished without assistant message")
	}

	toolCalls := extractToolCalls(assistantMsg)
	toolMsgs := executeTools(ctx, toolCalls, req.Tools, emitter)

	result := StepResult(append([]Message{assistantMsg}, toolMsgs...))
	cancelled := ctx.Err() != nil
	emitter.delta(StepStatusDelta{Cancelled: cancelled})

	if cancelled {
		return result, ctx.Err()
	}
	return result, nil
}

func handleProviderUpdate(up ProviderUpdate, emitter stepEmitter) (AssistantMessage, bool, error) {
	switch u := up.(type) {
	case nil:
		return AssistantMessage{}, false, nil
	case ProviderDeltaUpdate:
		if u.Delta != nil {
			emitter.delta(u.Delta)
		}
		return AssistantMessage{}, false, nil
	case ProviderMessageUpdate:
		emitter.message(u.Message)
		return u.Message, true, nil
	default:
		return AssistantMessage{}, false, errors.New("step: unknown provider update")
	}
}

func executeTools(ctx context.Context, calls []ToolCallPart, tools []Tool, emitter stepEmitter) []Message {
	if len(calls) == 0 {
		return nil
	}

	toolMap := map[string]Tool{}
	for _, t := range tools {
		spec := t.Spec()
		toolMap[spec.Name] = t
	}

	results := make([]ToolResult, len(calls))
	msgs := make([]Message, len(calls))
	completed := make([]bool, len(calls))

	toolCtx, cancelTools := context.WithCancel(ctx)
	defer cancelTools()

	// completions is buffered to avoid blocking tool goroutines when the step is cancelled.
	type completion struct {
		idx int
		res ToolResult
	}
	completions := make(chan completion, len(calls))
	parallelIdx := make([]bool, len(calls))

	execOne := func(idx int, call ToolCallPart) {
		emitter.delta(ToolExecDelta{CallID: call.CallID, Name: call.Name, Stage: ToolExecStart})
		res := executeSingleTool(toolCtx, call, toolMap)
		select {
		case completions <- completion{idx: idx, res: res}:
		default:
			// Drop if receiver stopped (e.g. cancelled), Step will emit interrupted results.
		}
	}

	flushInOrder := func(next *int) {
		for *next < len(calls) && completed[*next] {
			idx := *next
			if msgs[idx] != nil {
				*next = *next + 1
				continue
			}
			res := results[idx]
			msg := ToolResultMessage{
				CallID:    res.CallID,
				Name:      res.Name,
				IsError:   res.IsError,
				Parts:     res.Parts,
				Timestamp: time.Now().UnixMilli(),
				Details:   res.Details,
			}
			msgs[idx] = msg
			emitter.message(msg)
			emitter.delta(ToolExecDelta{CallID: res.CallID, Name: res.Name, Stage: ToolExecEnd})
			*next = *next + 1
		}
	}

	nextToEmit := 0

	recordCompletion := func(idx int, res ToolResult) {
		if idx < 0 || idx >= len(calls) {
			return
		}
		if completed[idx] {
			return
		}
		results[idx] = res
		completed[idx] = true
		flushInOrder(&nextToEmit)
	}

	// Execute tools with a simple exclusivity rule:
	// - tools with Spec().Parallel=true may run concurrently with each other
	// - tools with Spec().Parallel=false run exclusively
	var runningParallel int

	startParallel := func(idx int, call ToolCallPart) {
		runningParallel++
		parallelIdx[idx] = true
		go execOne(idx, call)
	}

	markInterruptedFrom := func(start int) {
		for i := start; i < len(calls); i++ {
			if completed[i] {
				continue
			}
			results[i] = interruptedToolResult(calls[i])
			completed[i] = true
		}
	}

	recvOne := func() bool {
		select {
		case <-ctx.Done():
			cancelTools()
			markInterruptedFrom(0)
			flushInOrder(&nextToEmit)
			return false
		case c := <-completions:
			recordCompletion(c.idx, c.res)
			if c.idx >= 0 && c.idx < len(parallelIdx) && parallelIdx[c.idx] {
				runningParallel--
			}
			return true
		}
	}

	for idx, call := range calls {
		if ctx.Err() != nil {
			markInterruptedFrom(idx)
			flushInOrder(&nextToEmit)
			break
		}

		tool, ok := toolMap[call.Name]
		parallel := ok && tool.Spec().Parallel

		if !parallel {
			// Wait for any parallel tools to finish before executing a non-parallel tool.
			for runningParallel > 0 {
				if !recvOne() {
					break
				}
			}
			if ctx.Err() != nil {
				recordCompletion(idx, interruptedToolResult(call))
				continue
			}
			emitter.delta(ToolExecDelta{CallID: call.CallID, Name: call.Name, Stage: ToolExecStart})
			res := executeSingleTool(toolCtx, call, toolMap)
			recordCompletion(idx, res)
			continue
		}

		startParallel(idx, call)
	}

	for runningParallel > 0 {
		if !recvOne() {
			break
		}
	}

	// Ensure every tool call has a result message.
	for i := range calls {
		if completed[i] {
			continue
		}
		recordCompletion(i, interruptedToolResult(calls[i]))
	}

	flushInOrder(&nextToEmit)
	return msgs
}

type stepEmitter struct {
	onDelta   func(MessageDelta)
	onMessage func(Message)
}

func (e stepEmitter) delta(d MessageDelta) {
	if d == nil || e.onDelta == nil {
		return
	}
	e.onDelta(d)
}

func (e stepEmitter) message(m Message) {
	if m == nil || e.onMessage == nil {
		return
	}
	e.onMessage(m)
}

func executeSingleTool(ctx context.Context, call ToolCallPart, toolMap map[string]Tool) ToolResult {
	if ctx.Err() != nil {
		return interruptedToolResult(call)
	}
	tool, ok := toolMap[call.Name]
	if !ok {
		return toolNotFoundResult(call)
	}

	res, err := tool.Execute(ctx, call)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return interruptedToolResult(call)
		}
		return errorToolResult(call, err)
	}
	if res.CallID == "" {
		res.CallID = call.CallID
	}
	if res.Name == "" {
		res.Name = call.Name
	}
	return res
}

func interruptedToolResult(call ToolCallPart) ToolResult {
	return ToolResult{
		CallID:  call.CallID,
		Name:    call.Name,
		IsError: true,
		Parts:   []Part{TextPart{Text: "Request interrupted by user for tool use"}},
	}
}

func collectToolSpecs(tools []Tool) []ToolSpec {
	specs := make([]ToolSpec, 0, len(tools))
	for _, t := range tools {
		specs = append(specs, t.Spec())
	}
	return specs
}

func extractToolCalls(msg AssistantMessage) []ToolCallPart {
	var calls []ToolCallPart
	for _, part := range msg.Parts {
		tc, ok := part.(ToolCallPart)
		if !ok {
			continue
		}
		calls = append(calls, tc)
	}
	return calls
}

func toolNotFoundResult(call ToolCallPart) ToolResult {
	return ToolResult{
		CallID:  call.CallID,
		Name:    call.Name,
		IsError: true,
		Parts:   []Part{TextPart{Text: "tool not found"}},
	}
}

func errorToolResult(call ToolCallPart, err error) ToolResult {
	return ToolResult{
		CallID:  call.CallID,
		Name:    call.Name,
		IsError: true,
		Parts:   []Part{TextPart{Text: err.Error()}},
	}
}
