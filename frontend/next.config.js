/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'standalone',
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || (process.env.NODE_ENV === 'production' ? 'https://production-rag-api-nd1p.onrender.com' : 'http://localhost:8000'),
    NEXT_PUBLIC_API_KEY: process.env.NEXT_PUBLIC_API_KEY || 'rag-prod-secret-key-change-me',
  }
};

module.exports = nextConfig;
