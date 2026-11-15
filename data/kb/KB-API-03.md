---
id: KB-API-03
title: Troubleshooting API errors
tags: api, technical
---

HTTP 401: the token is invalid or revoked. HTTP 403: the token's user lacks permission on the resource. HTTP 429: see rate limits. HTTP 500 or 503: a server-side error; check the status page and send the request id (X-Request-Id header) to support so engineering can trace the call.
