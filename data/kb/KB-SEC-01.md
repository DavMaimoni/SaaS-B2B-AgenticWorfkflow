---
id: KB-SEC-01
title: Single sign-on (SAML) and SCIM
tags: security, it
---

SSO with SAML 2.0 (Okta, Azure AD, Google Workspace, OneLogin) and SCIM 2.0 provisioning are available on the Enterprise plan, in Settings > Security > SSO.
Setup: copy the ACS URL and Entity ID into the identity provider, then upload its metadata XML. If every user suddenly gets a SAML error, the identity provider's signing certificate has usually expired: upload the new metadata. Admins can always sign in with email and password plus 2FA as a break-glass access.
