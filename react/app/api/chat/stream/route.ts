import { NextRequest } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const STREAM_HEADERS: Record<string, string> = {
  'Content-Type': 'text/event-stream; charset=utf-8',
  'Cache-Control': 'no-cache, no-transform',
  Connection: 'keep-alive',
  'X-Accel-Buffering': 'no',
};

function sseErrorResponse(message: string, status: number): Response {
  const payload = JSON.stringify({ event: 'error', error: message });
  return new Response(`event: error\ndata: ${payload}\n\ndata: [DONE]\n\n`, {
    status,
    headers: STREAM_HEADERS,
  });
}

export async function POST(request: NextRequest) {
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

    const safeClose = () => {
      if (closed) return;
      closed = true;
      if (streamController) {
        try {
          streamController.close();
        } catch {}
      }
    };

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
