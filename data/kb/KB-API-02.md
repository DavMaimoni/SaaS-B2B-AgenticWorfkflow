---
id: KB-API-02
title: API tokens and pagination
tags: api, technical
---

Tokens are created and revoked in Settings > API. To rotate without downtime: create a new token, deploy it, then revoke the old one; both work until revocation.
List endpoints use cursor-based pagination: pass the next_cursor value of the previous response; an empty next_cursor means the last page. Page size is 100 items by default, 500 at most.
