import { NextRequest } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

// 统一 SSE 头，尽量降低中间层缓存/缓冲对流式实时性的影响。
const STREAM_HEADERS: Record<string, string> = {
  'Content-Type': 'text/event-stream; charset=utf-8',
  'Cache-Control': 'no-cache, no-transform',
  Connection: 'keep-alive',
  'X-Accel-Buffering': 'no',
};

function sseErrorResponse(message: string, status: number): Response {
  // 代理层错误也保持 SSE 协议，前端复用同一解析逻辑。
  const payload = JSON.stringify({ event: 'error', error: message });
  return new Response(`event: error\ndata: ${payload}\n\ndata: [DONE]\n\n`, {
    status,
    headers: STREAM_HEADERS,
  });
}

export async function POST(request: NextRequest) {
  // 读取原始 body 直传后端，避免代理层改写请求结构。
  const body = await request.text();
  const backendCandidates = [
    process.env.BACKEND_API_BASE?.trim(),
    'http://127.0.0.1:8001',
    'http://127.0.0.1:8000',
  ].filter((v): v is string => Boolean(v));
  const upstreamAbort = new AbortController();
  const onClientAbort = () => upstreamAbort.abort();
  request.signal.addEventListener('abort', onClientAbort);

  try {
    let response: Response | null = null;
    for (const base of backendCandidates) {
      try {
        response = await fetch(`${base}/api/chat/stream`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json; charset=utf-8',
          },
          body: body,
          signal: upstreamAbort.signal,
        });
        if (response.ok) {
          break;
        }
      } catch {
        response = null;
      }
    }

    if (!response) {
      return sseErrorResponse('Backend unreachable', 502);
    }

    if (!response.ok) {
      return sseErrorResponse('Backend error', response.status);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      return sseErrorResponse('Backend stream unavailable', 502);
    }
    let streamController: ReadableStreamDefaultController<Uint8Array> | null = null;
    let closed = false;
    let readerCancelled = false;

    // 单路径关闭，避免 cancel/finally 并发触发导致二次 close。
    const safeClose = () => {
      if (closed) return;
      closed = true;
      if (streamController) {
        try {
          streamController.close();
        } catch {}
      }
    };

    // 上游 reader 取消也做幂等，避免噪声异常。
    const safeCancelReader = async () => {
      if (readerCancelled) return;
      readerCancelled = true;
      try {
        await reader.cancel();
      } catch {}
    };

    const stream = new ReadableStream({
      async start(controller) {
        streamController = controller;
        try {
          // 直接透传后端 chunk，不做解析重组，减少代理层开销。
          while (true) {
            if (upstreamAbort.signal.aborted || closed) break;
            const { done, value } = await reader.read();
            if (done) break;
            if (closed) break;
            controller.enqueue(value);
          }
        } catch {
          // 上游中断或客户端取消时直接结束流
        } finally {
          safeClose();
          await safeCancelReader();
        }
      },
      async cancel() {
        // 当前端主动停止时，把取消信号传递到上游后端链路。
        upstreamAbort.abort();
        safeClose();
        await safeCancelReader();
      },
    });

    return new Response(stream, {
      status: response.status,
      headers: STREAM_HEADERS,
    });
  } catch {
    if (upstreamAbort.signal.aborted) {
      return sseErrorResponse('Client cancelled', 499);
    }
    return sseErrorResponse('Proxy error', 500);
  } finally {
    request.signal.removeEventListener('abort', onClientAbort);
  }
}
