/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        // 排除 /api/chat/stream，让新的 API Route 处理
        source: '/api/:path*',
        destination: 'http://127.0.0.1:8001/api/:path*',
      },
      {
        source: '/app',
        destination: 'http://127.0.0.1:8001/app',
      },
      {
        source: '/static/:path*',
        destination: 'http://127.0.0.1:8001/static/:path*',
      },
    ];
  },
}

module.exports = nextConfig
