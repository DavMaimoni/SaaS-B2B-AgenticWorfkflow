---
id: KB-API-01
title: API rate limits
tags: api, technical
---

The REST API is limited per workspace: Free 60 requests per minute, Pro 300 requests per minute, Enterprise 1,000 requests per minute.
Above the limit the API answers HTTP 429 Too Many Requests with a Retry-After header (seconds): wait that long before retrying, with exponential backoff. Batch endpoints (/v1/batch) accept up to 100 operations per call and count as one request.
