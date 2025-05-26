import httpx


def get_code_server_cookies():
    cookies = httpx.Cookies()
    cookies.set(
        "code-server-session",
        "%24argon2id%24v%3D19%24m%3D65536%2Ct%3D3%2Cp%3D4%24ZccjUX4eBTY6N2BBDDUhsA%24yUfyP7nM1GFTD%2Fj%2FO6gFll5n4Fzm2olxz4aVuC28UI0",
        domain=".4cdeb7c.tunnel.myubai.uos.ac.kr",
        path="/"
    )
    return cookies 