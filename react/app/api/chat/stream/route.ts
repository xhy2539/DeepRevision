import { NextRequest } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
  const body = await request.text();
  const backendCandidates = [
    process.env.BACKEND_API_BASE?.trim(),
    'http://127.0.0.1:8001',
    'http://127.0.0.1:8000',
  ].filter((v): v is string => Boolean(v));

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
        });
        if (response.ok) {
          break;
        }
      } catch {
        response = null;
      }
    }

    if (!response) {
      return new Response(`data: {"error": "Backend unreachable"}\n\n`, {
        status: 502,
        headers: { 'Content-Type': 'text/event-stream' },
      });
    }

    if (!response.ok) {
      return new Response(`data: {"error": "Backend error"}\n\n`, {
        status: response.status,
        headers: { 'Content-Type': 'text/event-stream' },
      });
    }

    // 直接转发流，不做处理
    const stream = new ReadableStream({
      async start(controller) {
        let isClosed = false;
        const reader = response.body?.getReader();
        if (!reader) {
          controller.close();
          return;
        }
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            controller.enqueue(value);
          }
        } catch (e) {
          // 流读取中断，但前端会收到 [DONE] 终止
        } finally {
          if (!isClosed) {
            controller.close();
            isClosed = true;
          }
        }
      },
    });

    return new Response(stream, {
      status: response.status,
      headers: {
        'Content-Type': 'text/event-stream',
        'Transfer-Encoding': 'chunked',
      },
    });
  } catch (error) {
    return new Response(`data: {"error": "Proxy error"}\n\n`, {
      status: 500,
      headers: { 'Content-Type': 'text/event-stream' },
    });
  }
}
