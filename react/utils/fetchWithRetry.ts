export async function fetchWithRetry(
  url: string,
  options: RequestInit = {},
  retries: number = 2,
  delay: number = 1000
): Promise<Response> {
  let lastError: Error | null = null;

  for (let i = 0; i <= retries; i++) {
    try {
      const response = await fetch(url, options);
      if (response.ok || i === retries) {
        return response;
      }
      lastError = new Error(`HTTP ${response.status}`);
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
    }

    if (i < retries) {
      await new Promise(resolve => setTimeout(resolve, delay * (i + 1)));
    }
  }

  throw lastError || new Error("Fetch failed");
}
